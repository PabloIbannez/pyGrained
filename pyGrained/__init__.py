import sys,os
import gc

from tqdm import tqdm

import uuid

import pathlib

import numpy as np

from Bio.PDB import *

from scipy.spatial.transform import Rotation
from string import ascii_uppercase

import logging
import warnings

import pdb

#############################################################

from .utils.sequence import *
from .utils.SASA     import *

class CustomFormatter(logging.Formatter):

    white = "\x1b[37;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format     = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
    formatLine = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: white + formatLine + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + formatLine + reset,
        logging.CRITICAL: bold_red + formatLine + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt,datefmt='%d/%m/%Y %H:%M:%S')
        return formatter.format(record)

#############################################################
####################### SET UP LOGGER #######################

# Set up logger

logger = logging.getLogger(f"pyGrained")
logger.setLevel(logging.DEBUG)

clog = logging.StreamHandler()
clog.setLevel(logging.DEBUG)

clog.setFormatter(CustomFormatter())
logger.addHandler(clog)

class CoarseGrainedBase:

    def __getClasses(self,structure):
        """
        It groups the chains of the structure in classes.
        For every chain its sequence is computed.
        Two chains are in the same class if the sequence is the same
        or if the sequence of one chain is a subsequence of the other.
        For every class a leader is chosen. The leader is the chain
        with the smallest sequence.

        Returns a dictionary with the following structure:
        {
            "class1": {
                "leader": "A",
                "members": ["A","B","C"]
            },
            "class2": {
                "leader": "D",
                "members": ["D","E"]
            }
        }

        Note that a chain can belong to a one class only.
        Note that leaders are also considered members of the class.
        """

        self.logger.info(f"Assigning classes ...")

        #######################

        m0 = list(structure.get_models())[0]

        chains = []
        ch2seq = {}
        chLen  = {}
        for ch in m0.get_chains():
            name = ch.get_id()
            seq  = getChainSequence(ch).strip('X')
            chains.append(name)
            ch2seq[name] = seq
            chLen[name]  = len(seq)

        chContainsCh = {}
        for ch in chains:
            chContainsCh[ch] = []
            for ch_toCheck in chains:
                if ch != ch_toCheck:
                    if ch2seq[ch_toCheck] in ch2seq[ch]:
                        chContainsCh[ch].append(ch_toCheck)

        classes = {}
        for ch in chains:
            newClass = True
            for className in classes.keys():
                for mmb in classes[className]["members"]:
                    if ch in mmb or mmb in chContainsCh[ch]:
                        classes[className]["members"].update(ch)
                        classes[className]["members"].update(chContainsCh[ch])
                        newClass = False
                        break
            if newClass:
                name = ascii_uppercase[len(classes.keys())]
                classes[name] = {"leader":[],"members":set(chContainsCh[ch]+[ch])}

        for clName,info in classes.items():
            info["members"] = list(info["members"])

        for clName,info in classes.items():
            leaderLen = chLen[list(info["members"])[0]]
            info["leader"] = list(info["members"])[0]
            for mmb in info["members"]:
                if chLen[mmb] < leaderLen:
                    leaderLen = chLen[mmb]
                    info["leader"] = mmb

        for clName,info in classes.items():
            leader  = info["leader"]
            members = info["members"]
            self.logger.info(f"Class {clName} which leader is {leader} has the following members: {members}")

        return classes

    def __aggregateStructure(self,structure,classes):

        m0 = list(structure.get_models())[0]
        aggregatedStructure = Structure.Structure(structure.get_id()+"_aggregated")

        atomCount = 1

        mdl_agg = Model.Model(m0.get_id())
        aggregatedStructure.add(mdl_agg)
        for ch in m0.get_chains():
            for clsName,info in classes.items():
                chName = info["leader"]
                if ch.get_id() == chName:
                    ch_agg = Chain.Chain(chName)
                    mdl_agg.add(ch_agg)
                    for res in ch.get_residues():
                        res_agg = Residue.Residue(res.get_id(),res.get_resname(),res.get_segid())
                        ch_agg.add(res_agg)

                        for atm in res.get_atoms():
                            atm_agg = Atom.Atom(atm.get_name(),
                                                atm.get_coord(),
                                                atm.get_bfactor(),
                                                atm.get_occupancy(),
                                                atm.get_altloc(),
                                                atm.get_fullname(),
                                                atomCount,
                                                element=atm.element);

                            chrg = atm.get_charge()
                            if chrg == None:
                                atm_agg.set_charge(0.0)
                            else:
                                atm_agg.set_charge(chrg)

                            atm_agg.mass   = atm.mass
                            atm_agg.radius = atm.radius
                            if hasattr(atm,'totalSASA'):
                                atm_agg.totalSASA = atm.totalSASA
                            if hasattr(atm,'totalSASApolar'):
                                atm_agg.totalSASApolar = atm.totalSASApolar
                            if hasattr(atm,'totalSASAapolar'):
                                atm_agg.totalSASAapolar = atm.totalSASAapolar

                            res_agg.add(atm_agg)
                            atomCount+=1

        return aggregatedStructure

    def __computeTransformations(self,structure,aggregatedStructure,classes):

        transformations = {}

        #########################################

        for clsName in classes.keys():

            for ch_prc in aggregatedStructure.get_chains():
                chName = classes[clsName]["leader"]
                if ch_prc.get_id() == chName:
                    reference = list(ch_prc.get_atoms())
                    minimun,maximun = [ch_prc.child_list[0].get_id()[1],ch_prc.child_list[-1].get_id()[1]]

            transformations[clsName] = []
            for m in structure.get_models():
                for i,ch in enumerate(m.get_chains()):
                    if ch.get_id() in classes[clsName]["members"]:

                        mobile    = list(ch.get_atoms())

                        referencePos = np.asarray([atm.get_coord() for atm in reference if (atm.get_name() == "CA" and \
                                                                                            atm.get_parent().get_id()[1] >= minimun and \
                                                                                            atm.get_parent().get_id()[1] <= maximun )])

                        mobilePos    = np.asarray([atm.get_coord() for atm in mobile if (atm.get_name() == "CA" and \
                                                                                         atm.get_parent().get_id()[1] >= minimun and \
                                                                                         atm.get_parent().get_id()[1] <= maximun )])

                        #Aling mobile with reference

                        #Translation
                        trans = np.mean(mobilePos,axis=0) - np.mean(referencePos,axis=0)

                        #Move mobile and reference to origin
                        referencePos = referencePos - np.mean(referencePos,axis=0)
                        mobilePos    = mobilePos - np.mean(mobilePos,axis=0)

                        #Rotation over mobile to get reference
                        rot = Rotation.align_vectors(mobilePos,referencePos)[0]

                        transformations[clsName].append([m.get_id(),ch.get_id(),trans,rot])

        return transformations

    def __spreadStructure(self,structure,classes):

        mdlCh2transform = {}
        for clsName in classes.keys():
            for mdl,ch,t,r in classes[clsName]["transformations"]:
                if mdl not in mdlCh2transform.keys():
                    mdlCh2transform[mdl] = {}
                if ch in mdlCh2transform[mdl].keys():
                    self.logger.critical(f"Chain has been added before for model: {mdl}")
                    raise Exception(f"Chain has been added before for model")

                chName = classes[clsName]["leader"]
                mdlCh2transform[mdl][ch] = [chName,t,r]

        models = set()
        chains = set()
        for clsName in classes.keys():
            for mdl,ch,_,_ in classes[clsName]["transformations"]:
                models.add(mdl)
                chains.add(ch)

        spreadedStructure = Structure.Structure(structure.get_id()+"_spreaded")

        atomCount = 1
        for mdl_id in models:
            mdl = Model.Model(mdl_id)
            spreadedStructure.add(mdl)
            for ch_id in chains:
                ch = Chain.Chain(ch_id)
                mdl.add(ch)

                prcChName,t,r = mdlCh2transform[mdl_id][ch_id]

                referenceCh = None
                for prcCh in structure.get_chains():
                    if prcChName == prcCh.get_id():
                        referenceCh = prcCh
                        break

                if referenceCh == None:
                    self.logger.debug(f"No chain found in aggregated structure for : ({mdl_id} {ch_id})")
                else:
                    for res_ref in referenceCh.get_residues():
                        res = Residue.Residue(res_ref.get_id(),res_ref.get_resname(),res_ref.get_segid())
                        ch.add(res)

                        for atm_ref in res_ref.get_atoms():

                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')

                                atm = Atom.Atom(atm_ref.get_name(),
                                                atm_ref.get_coord(),
                                                atm_ref.get_bfactor(),
                                                atm_ref.get_occupancy(),
                                                atm_ref.get_altloc(),
                                                atm_ref.get_fullname(),
                                                atomCount,
                                                element=atm_ref.element);

                                atm.set_charge(atm_ref.get_charge())
                                atm.mass   = atm_ref.mass
                                atm.radius = atm_ref.radius

                                if hasattr(atm_ref,'totalSASA'):
                                    atm.totalSASA = atm_ref.totalSASA
                                if hasattr(atm_ref,'totalSASApolar'):
                                    atm.totalSASApolar = atm_ref.totalSASApolar
                                if hasattr(atm_ref,'totalSASAapolar'):
                                    atm.totalSASAapolar = atm_ref.totalSASAapolar

                                res.add(atm)
                            atomCount+=1

                    referencePositions = np.asarray([ atm.get_coord() for atm in referenceCh.get_atoms()])
                    ref2orig = np.mean(referencePositions,axis=0)
                    referencePositions = referencePositions - ref2orig

                    mobilePositions = r.apply(referencePositions)
                    mobilePositions = mobilePositions + ref2orig
                    mobilePositions = mobilePositions + t

                    for i,atm_mobile in enumerate(ch.get_atoms()):
                        atm_mobile.set_coord(mobilePositions[i])

        for index,atm in enumerate(spreadedStructure.get_atoms()):
            atm.set_serial_number(index)

        return spreadedStructure

    def __init__(self,
                 tpy:str,name:str,
                 inputPDBfilePath:str,
                 fixPDB = False,
                 removeHetatm = True, removeHydrogens = True,removeNucleics  = True,
                 centerInput = True,
                 SASA = False,
                 aggregateChains = True,
                 debug = False):

        self._type = tpy
        self._name = name

        self.logger = logging.getLogger(f"pyGrained")

        #Set all loggers level
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        #############################################################
        ##################### TRY LOAD PDBFIXER #####################

        if fixPDB:

            try:

                from pdbfixer import PDBFixer
                from openmm.app import PDBFile

            except:

                self.logger.warning(f"PDBFixer not found, fixPDB will be ignored")
                fixPDB = False


        #############################################################
        ####################### LOADING INPUT #######################

        #Load PDB
        self.logger.info(f"Loadding {inputPDBfilePath} ...")

        file_extension = pathlib.Path(inputPDBfilePath).suffix
        if   (file_extension == ".pdb"):

            inputStructure = PDBParser().get_structure(name,inputPDBfilePath)

            if fixPDB:

                self.logger.info("Fixing PDB file ...")

                #Write each model in inputStructure to file
                #Fix that file and then update model in inputStructure

                class ModelFixer(Select):

                    def __init__(self,mdlId):
                        super().__init__()
                        self.mdlId = mdlId

                    def accept_model(self, model):
                        if model.get_id() == self.mdlId:
                            return True
                        else:
                            return False

                for mdl in tqdm(list(inputStructure.get_models())):

                    mdlId = mdl.get_id()
                    mdlFixer = ModelFixer(mdlId)

                    io = PDBIO()
                    io.set_structure(inputStructure)
                    io.save(f"tmp.pdb",mdlFixer)

                    ############ FIX TMP.PDB ############

                    fixer = PDBFixer(filename = "tmp.pdb")

                    fixer.missingResidues = {}

                    fixer.findMissingAtoms()
                    fixer.addMissingAtoms()
                    fixer.addMissingHydrogens(7.0)

                    PDBFile.writeFile(fixer.topology, fixer.positions, open("fixed.pdb", 'w'), keepIds=True)

                    ############ UPDATE MODEL ############

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mdlFixed = list(PDBParser().get_structure(name,"fixed.pdb").get_models())[0]

                    mdlFixed.id = mdlId

                    struct = mdl.get_parent()

                    struct.detach_child(mdl.get_id())
                    mdl.detach_parent()

                    struct.add(mdlFixed)
                    mdlFixed.set_parent(struct)

                    #Remove aux files
                    os.remove("tmp.pdb")
                    os.remove("fixed.pdb")

                io = PDBIO()
                io.set_structure(inputStructure)
                io.save(inputPDBfilePath.replace(".pdb","_fixed.pdb"))

                self.logger.info("PDB fixed")

            chargeInInput = False
        elif (file_extension == ".pqr"):
            inputStructure = PDBParser(is_pqr=True).get_structure(name,inputPDBfilePath)
            chargeInInput = True
        else:
            self.logger.critical(f"The extension {file_extension} of the input file {inputPDBfilePath} can not be handle")
            raise Exception(f"Not supported file extension")

        self.logger.info(f"Input file {inputPDBfilePath} has been loaded.")

        #############################################################
        ########################## REMOVE ###########################

        if removeHetatm:
            self.logger.info("Removing HETATM ...")
            res2remove = []
            chains     = list(inputStructure.get_chains())
            for i,ch in enumerate(chains):
                for res in ch:
                    hetKey = res.get_full_id()[3][0]
                    if hetKey != " ":
                        res2remove.append([i,res.id])

            for i,res_id in res2remove:
                chains[i].detach_child(res_id)

        if removeHydrogens:
            self.logger.info(f"Removing hydrogens ...")
            atm2remove = []
            residues   = list(inputStructure.get_residues())
            for i,res in enumerate(residues):
                for atm in res.get_atoms():
                    element = atm.element
                    if element == "H":
                        atm2remove.append([i,atm.id])

            for i,atm_id in atm2remove:
                residues[i].detach_child(atm_id)

        if removeNucleics:
            self.logger.info(f"Removing nucleics ...")
            nucleics = ['A','C','G','U','I','T','DA','DC','DG','DT','DI','DU']
            ch2remove = set()
            models = list(inputStructure.get_models())
            for i,md in enumerate(models):
                for res in md.get_residues():
                    if res.get_resname().strip() in nucleics:
                        ch2remove.add((i,res.get_parent().get_id()))

            for i,ch_id in ch2remove:
                models[i].detach_child(ch_id)

        #############################################################
        ########################## CENTER ###########################

        if centerInput:
            self.logger.info(f"Centering input structure ...")
            center = np.asarray([atm.get_coord() for atm in inputStructure.get_atoms()]).mean(axis=0)
            for atm in inputStructure.get_atoms():
                atm.set_coord(atm.get_coord()-center)

        #############################################################
        ########################### SASA ############################

        if SASA:
            self.logger.info(f"Computing SASA ...")
            sasa = computeStructureSASA(inputStructure)
            gc.collect()

            for i,atm in enumerate(inputStructure.get_atoms()):
                atm.totalSASA = sasa[i]

        #############################################################
        #################### CLASSES DEFINITION #####################

        if aggregateChains:
            classes = self.__getClasses(inputStructure);
        else:
            #If chains are not aggregated, each chain is a class

            m0  = list(inputStructure.get_models())[0]

            classes = {}
            for ch in m0.get_chains():
                chName = ch.get_id()
                classes[chName] = {"leader":chName,"members":chName}

        #Note than we only work with the first model.
        #It is because models are assumed to be the same
        #e.g. every model contains exactly the same atoms but in different positions.
        #This is expected for regular PDBs. If this is not the case, the code will fail.

        #############################################################
        #################### AGGREAGATE STRUCTURE ###################

        self.logger.info(f"Aggregating structure ...")
        aggregatedStructure = self.__aggregateStructure(inputStructure,classes)

        #aggregateStructure contains only one model.
        #This model is made of the leader chain of every class.

        #############################################################
        ################## COMPUTE TRANSFORMATIONS ##################

        #The transformation (translation and rotation) for each chain is computed

        self.logger.info(f"Computing transformations ...")
        transformations = self.__computeTransformations(inputStructure,aggregatedStructure,classes)

        #Transformations are stored in the classes dictionary
        for name,transf in transformations.items():
            classes[name]["transformations"]=transf.copy()

        #Each transformation stores the translation and rotation
        #to be applied over the leader chain to obtain the member chain,
        #for the different conformations of the member chain (different models),
        for cls in transformations.keys():
            for mdl,ch,t,r in classes[name]["transformations"]:
                self.logger.debug(f"Transformation {cls} : {mdl} {ch} {t} {r}")

        #############################################################
        ###################### SPREAD STRUCTURE #####################

        #The transformations are applied to the aggregated structure
        #to obtain the spreaded structure. This structure is similar
        #to the input structure but we are using the leader chains.
        #This ensures that all the chains (of the same class)
        #have the same number of atoms and are equivalent
        #across the different conformations (models).

        #Note that spreaded structure is the structure we should work with.
        #It is prepared for that. For example for a spread structured particle
        #indices are ensured to start with 0 and be consecutive

        self.logger.info(f"Spreading structure ...")
        spreadedStructure = self.__spreadStructure(aggregatedStructure,classes)

        #############################################################

        self.logger.info(f"Initialization end.")

        #We have defined the following attributes:

        #chargeInInput: True if the input file contains charges
        #inputStructure: the input structure
        #aggregatedStructure: the aggregated structure
        #spreadedStructure: the spreaded structure
        #classes: the classes of chains

        self._chargeInInput       = None
        self._inputStructure      = None
        self._aggregatedStructure = None
        self._spreadedStructure   = None
        self._classes             = None

        self.setChargeInInput(chargeInInput)
        self.setInputStructure(inputStructure)
        self.setAggregatedStructure(aggregatedStructure)
        self.setSpreadedStructure(spreadedStructure)
        self.setClasses(classes)

        #############################################################

        #We define the attributes that has to be added by the derived class

        self._aggregatedCgStructure = None
        self._spreadedCgStructure   = None

        self._aggregatedCgMap = None
        self._spreadedCgMap   = None

        self._types      = None
        self._states     = None
        self._structure  = None
        self._forceField = None

    ########################################################

    def setChargeInInput(self,chargeInInput):
        self._chargeInInput = chargeInInput

    def setInputStructure(self,inputStructure):
        self._inputStructure = inputStructure

    def setAggregatedStructure(self,aggregatedStructure):
        self._aggregatedStructure = aggregatedStructure

    def setSpreadedStructure(self,spreadedStructure):
        self._spreadedStructure = spreadedStructure

    def setClasses(self,classes):
        self._classes = classes

    def getChargeInInput(self):
        if self._chargeInInput is None:
            self.logger.error(f"Charge in input not set.")
            raise ValueError("chargeInInput not set.")
        return self._chargeInInput

    def getInputStructure(self):
        if self._inputStructure is None:
            self.logger.error(f"Input structure not set.")
            raise ValueError("inputStructure not set.")
        return self._inputStructure

    def getAggregatedStructure(self):
        if self._aggregatedStructure is None:
            self.logger.error(f"Aggregated structure not set.")
            raise ValueError("aggregatedStructure not set.")
        return self._aggregatedStructure

    def getSpreadedStructure(self):
        if self._spreadedStructure is None:
            self.logger.error(f"Spreaded structure not set.")
            raise ValueError("spreadedStructure not set.")
        return self._spreadedStructure

    def getClasses(self):
        if self._classes is None:
            self.logger.error(f"Classes not set.")
            raise ValueError("classes not set.")
        return self._classes

    ########################################################

    def setAggregatedCgStructure(self,aggregatedCgStructure):
        self._aggregatedCgStructure = aggregatedCgStructure

    def setSpreadedCgStructure(self,spreadedCgStructure):
        self._spreadedCgStructure = spreadedCgStructure

    def setAggregatedCgMap(self,aggregatedCgMap):
        self._aggregatedCgMap = aggregatedCgMap

    def setSpreadedCgMap(self,spreadedCgMap):
        self._spreadedCgMap = spreadedCgMap

    def getAggregatedCgStructure(self):
        if self._aggregatedCgStructure is None:
            self.logger.error(f"Aggregated CG structure not set.")
            raise ValueError("aggregatedCgStructure not set.")
        return self._aggregatedCgStructure

    def getSpreadedCgStructure(self):
        if self._spreadedCgStructure is None:
            self.logger.error(f"Spreaded CG structure not set.")
            raise ValueError("spreadedCgStructure not set.")
        return self._spreadedCgStructure

    def getAggregatedCgMap(self):
        if self._aggregatedCgMap is None:
            self.logger.error(f"Aggregated CG map not set.")
            raise ValueError("aggregatedCgMap not set.")
        return self._aggregatedCgMap

    def getSpreadedCgMap(self):
        if self._spreadedCgMap is None:
            self.logger.error(f"Spreaded CG map not set.")
            raise ValueError("spreadedCgMap not set.")
        return self._spreadedCgMap

    ########################################################

    def setTypes(self,types):
        self._types = types

    def setState(self,state):
        self._state = state

    def setStructure(self,structure):
        self._structure = structure

    def setForceField(self,forceField):
        self._forceField = forceField

    def getTypes(self):
        if self._types is None:
            self.logger.error(f"Types not set")
            raise Exception(f"Types not set")
        return self._types

    def getState(self):
        if self._state is None:
            self.logger.error(f"State not set")
            raise Exception(f"State not set")
        return self._state

    def getStructure(self):
        if self._structure is None:
            self.logger.error(f"Structure not set")
            raise Exception(f"Structure not set")
        return self._structure

    def getForceField(self):
        if self._forceField is None:
            self.logger.error(f"Force field not set")
            raise Exception(f"Force field not set")
        return self._forceField
