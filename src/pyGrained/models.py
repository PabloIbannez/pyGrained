#import pdb
import sys
import gc

import json
import jsbeautifier

import pathlib

import numpy as np
import random

from pyGrained import aminoacidCharges
from pyGrained import computeK

from tqdm import tqdm

import itertools

from Bio.PDB import *
from Bio.Data.SCOPData import protein_letters_3to1

import freesasa

from scipy.spatial.transform import Rotation

from scipy.spatial import cKDTree

from string import ascii_uppercase

from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

import MDAnalysis as mda

import logging
import warnings

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


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

class CoarseGrainedBase:

    def __computeAtomListCOM(self,atmList):
        masses = []
        wPos   = []
        for atm in atmList:
            masses.append(atm.mass)
            wPos.append(atm.get_coord()*atm.mass)
        masses = np.asarray(masses)
        wPos   = np.asarray(wPos)

        totalMass = np.sum(masses)

        return np.sum(wPos,axis=0)/totalMass

    def __computeAtomListMass(self,atmList):
        masses = []
        for atm in atmList:
            masses.append(atm.mass)
        masses = np.asarray(masses)

        return np.sum(masses)

    def __computeAtomListCharge(self,atmList):
        charges = []
        for atm in atmList:
            chg = atm.get_charge()
            if chg == None:
                chg=0.0
            charges.append(chg)
        charges = np.asarray(charges)

        return np.sum(charges)

    def __computeAtomListChargeFromResidues(self,atmList):

        charge = 0.0
        for atm in atmList:
            if atm.get_name() == "CA":
                charge+=aminoacidCharges[atm.get_parent().get_resname()]

        return charge

    def __computeAtomListRadiusOfGyration(self,atmList):
        COM = self.__computeAtomListCOM(atmList)
        M   = self.__computeAtomListMass(atmList)

        wR2 = []
        for atm in atmList:
            R=np.linalg.norm(atm.get_coord()-COM)
            wR2.append(R*R*atm.mass)
        wR2 = np.asarray(wR2).sum()

        return np.sqrt(wR2/M)

    def __computeAtomListSASA(self,atmList):

        classifier = freesasa.Classifier()

        sasaPolar  = 0.0
        sasaApolar = 0.0

        for atm in atmList:
            tpy = classifier.classify(atm.get_parent().get_resname(),atm.get_name())
            if   tpy == "Polar":
                sasaPolar+=atm.totalSASA
            elif tpy == "Apolar":
                sasaApolar+=atm.totalSASA
            else:
                self.log.critical(f"The type of the atom ({atm.get_parent().get_resname()},{atm.get_name()} is {tpy}. But it has to be \"Polar\" or \"Apolar\")")
                sys.exit()

        return sasaPolar,sasaApolar

    def __computeStructureSASA(self,structure):

        sasa,_ = freesasa.calcBioPDB(structure)
        sasaAtom = [sasa.atomArea(i) for i in range(sasa.nAtoms())]

        #sasaAtom = [0 for atm in structure.get_atoms()]

        return sasaAtom

    def __getChainSeq(self,chain):
        seqDict = {res.get_id()[1]:protein_letters_3to1[res.get_resname()] for res in chain.get_residues()}

        seqId = sorted(list(seqDict.keys()))

        seq = ""
        for i in range(1,max(seqId)):
            if i in seqDict.keys():
                seq+=seqDict[i]
            else:
                seq+="X"

        return seq

    def __getClasses(self,structure):

        self.log.info(f"Assigning classes ...")

        #######################

        m0 = list(structure.get_models())[0]

        chains = []
        ch2seq = {}
        chLen  = {}
        for ch in m0.get_chains():
            name = ch.get_id()
            seq  = self.__getChainSeq(ch).strip('X')
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
            self.log.info(f"Class {clName} which leader is {leader} has the following members: {members}")

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

                        trans,rot = self.__alignSets(referencePos,mobilePos)
                        transformations[clsName].append([m.get_id(),ch.get_id(),trans,rot])
        return transformations


    def __alignSets(self,reference,mobile):
        #Translation
        trans = np.mean(mobile,axis=0) - np.mean(reference,axis=0)
        #Move mobile and reference to origin
        reference = reference - np.mean(reference,axis=0)
        mobile    = mobile - np.mean(mobile,axis=0)
        #Rotation over mobile to get reference
        R = Rotation.align_vectors(mobile,reference)[0]

        return trans,R

    def __spreadStructure(self,structure,classes):

        mdlCh2transform = {}
        for clsName in classes.keys():
            for mdl,ch,t,r in classes[clsName]["transformations"]:
                if mdl not in mdlCh2transform.keys():
                    mdlCh2transform[mdl] = {}
                if ch in mdlCh2transform[mdl].keys():
                    self.log.critical(f"Chain has been added before for model: {mdl}")
                    sys.exit()

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
                    self.log.debug(f"No chain found in aggregated structure for : ({mdl_id} {ch_id})")
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

    def writePDBcg(self,outName):
        self.log.info(f"Writing pdb")
        io=PDBIO(use_model_flag=1)
        io.set_structure(self.spreadedCG)
        io.save(outName)

    def writePQRcg(self,outName):
        self.log.info(f"Writing pqr")
        io=PDBIO(use_model_flag=1,is_pqr=True)
        io.set_structure(self.spreadedCG)
        io.save(outName)

    def writeSPcg(self,outName):
        self.log.info(f"Writing sp")
        with open(outName,"w") as f:
            for bead in self.spreadedCG.get_atoms():
                pos = bead.get_coord()
                r   = bead.radius
                c   = hash(bead.get_parent().get_parent().get_id())%256
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {r} {c}\n")

    def applyCoarseGrainedOverTrajectory(self,trajPDB,trajDCD):

        self.log.info(f"Applying coarse grained over trajectory ...")

        mdls = [bead[0] for bead in self.cgMap]
        m0   = min(mdls)
        m0CgMap = {bead:self.cgMap[bead] for bead in self.cgMap.keys() if bead[0] == m0}

        universe = mda.Universe(trajPDB,trajDCD)

        chains = [ch.segid for ch in universe.segments]

        cg2sel = {}
        for bead in m0CgMap.keys():
            ch_cg  = bead[1]
            res_cg = bead[2]
            atm_cg = bead[3]
            if ch_cg in chains:

                sel = "segid "+ ch_cg +" and ("
                for atm in m0CgMap[bead]:
                    resid = atm[2]
                    name  = atm[3]
                    index = atm[4]

                    sel += "("
                    sel += "resid " + str(resid) + " and name " + name
                    sel += ") or "
                sel = sel[:-3]
                sel += ")"

                cg2sel[(ch_cg,res_cg,atm_cg,index)] = universe.select_atoms(sel)

        cgTraj = {}
        for ts in tqdm(universe.trajectory):
            cgTraj[ts.frame] = {}
            for s in cg2sel.keys():
                cgPos = cg2sel[s].center_of_mass()

                cgTraj[ts.frame][s] = cgPos

        return cgTraj

    def __init__(self,name:str,inputPDBfilePath:str,removeHydrogens=True,SASA=False,aggregateChains=True,debug=False):

        if not hasattr(self,"objectName"):
            self.objectName = "CoarseGrainedBase"

        # Set up logger
        self.log = logging.getLogger(f"{self.objectName}:{id(self)}")
        self.log.setLevel(logging.DEBUG)

        self.clog = logging.StreamHandler()
        if debug:
            self.clog.setLevel(logging.DEBUG) #<----
        else:
            self.clog.setLevel(logging.INFO) #<----

        self.clog.setFormatter(CustomFormatter())
        self.log.addHandler(self.clog)

        #Load PDB
        self.log.info(f"Loadding {inputPDBfilePath} ...")

        file_extension = pathlib.Path(inputPDBfilePath).suffix
        if   (file_extension == ".pdb"):
            self.inputStructure = PDBParser().get_structure(name,inputPDBfilePath)
            self.chargeInInput = False
        elif (file_extension == ".pqr"):
            self.inputStructure = PDBParser(is_pqr=True).get_structure(name,inputPDBfilePath)
            self.chargeInInput = True
        else:
            self.log.critical(f"The extension {file_extension} of the input file {inputPDBfilePath} can not be handle")
            sys.exit()

        self.log.info(f"Input file {inputPDBfilePath} has been loaded.")
        if removeHydrogens:
            self.log.info(f"Removing hydrogens ...")
            atm2remove = []
            residues   = list(self.inputStructure.get_residues())
            for i,res in enumerate(residues):
                for atm in res.get_atoms():
                    element = atm.element
                    if element == "H":
                        atm2remove.append([i,atm.id])

            for i,atm_id in atm2remove:
                residues[i].detach_child(atm_id)

        #Center structure
        self.log.info(f"Centering structure.")
        center = np.asarray([atm.get_coord() for atm in self.inputStructure.get_atoms()]).mean(axis=0)
        for atm in self.inputStructure.get_atoms():
            atm.set_coord(atm.get_coord()-center)

        if SASA:
            self.log.info(f"Computing SASA.")
            sasa = self.__computeStructureSASA(self.inputStructure)
            gc.collect()

            for i,atm in enumerate(self.inputStructure.get_atoms()):
                atm.totalSASA = sasa[i]

        #########################################

        if aggregateChains:
            self.classes = self.__getClasses(self.inputStructure);
        else:
            m0  = list(self.inputStructure.get_models())[0]

            self.classes = {}
            for ch in m0.get_chains():
                chName = ch.get_id()
                self.classes[chName] = {"leader":chName,"members":chName}

        #########################################

        self.log.info(f"Aggregating structure ...")
        self.aggregatedStructure = self.__aggregateStructure(self.inputStructure,self.classes)

        #########################################
        self.log.info(f"Computing transformations ...")
        transformations = self.__computeTransformations(self.inputStructure,self.aggregatedStructure,self.classes)
        for name,transf in transformations.items():
            self.classes[name]["transformations"]=transf.copy()

        #########################################

        self.log.info(f"Spreading structure ...")
        self.spreadedStructure = self.__spreadStructure(self.aggregatedStructure,self.classes)

        #########################################

        self.log.info(f"Initialization end.")


class SBCG(CoarseGrainedBase):

    def __generateSBCG_raw(self,positions,weights,resolution,S,e0,eS,l0,lS,minBeads):

        Nall   = positions.shape[0]
        Nbeads = int(Nall/resolution)+1

        self.log.info(f"Generaiting SBCG, from {Nall} atoms to {Nbeads} beads")

        if Nbeads <= minBeads:
            return []

        ###########################

        rndIndices   = np.random.choice(Nall,size=Nbeads,replace=False)
        positions_cg = positions[rndIndices]

        ###########################

        indices = list(range(Nall))

        for s in tqdm(range(S)):
            index   = random.choices(indices,weights,k=1)[0]
            posAtom = positions[index]

            distances = cdist(positions_cg,posAtom.reshape(1,3))
            K = np.asarray(computeK.computeK(distances,Nbeads))

            epsilon = e0*np.power(eS/e0,float(s)/S)
            lambda_ = l0*Nbeads*np.power(lS/(l0*Nbeads),float(s)/S)

            positions_cg = positions_cg + (epsilon*np.exp(-K/lambda_)).reshape(-1,1)*(posAtom-positions_cg)

        return positions_cg

    def __generateENM(self,structure,cgMap,enmCut):

        atom2bead = {}
        chainsCg = set()
        #Invert map
        for bead,atomsList in cgMap.items():
            chId      = bead[1]
            chainsCg.add(chId) #Not all chains can be in the cg model

            beadIndex = bead[4]
            for atm in atomsList:
                atomIndex = atm[4]
                atom2bead[atomIndex] = beadIndex

        atomsCA      = [atm for atm in structure.get_atoms() if atm.get_name() == "CA"]
        atomsCACoord = np.asarray([atm.get_coord() for atm in structure.get_atoms() if atm.get_name() == "CA"])

        kd = cKDTree(atomsCACoord)
        bondCAAtoms = kd.query_pairs(enmCut)

        bondBeadsTmp = []
        for bnd in bondCAAtoms:

            mdl1Index = atomsCA[bnd[0]].get_parent().get_parent().get_parent().get_id()
            mdl2Index = atomsCA[bnd[1]].get_parent().get_parent().get_parent().get_id()

            ch1Index = atomsCA[bnd[0]].get_parent().get_parent().get_id()
            ch2Index = atomsCA[bnd[1]].get_parent().get_parent().get_id()

            if (ch1Index in chainsCg) and (ch2Index in chainsCg):
                if ch1Index == ch2Index and mdl1Index == mdl2Index:
                    bead1Index = atom2bead[atomsCA[bnd[0]].get_serial_number()]
                    bead2Index = atom2bead[atomsCA[bnd[1]].get_serial_number()]
                    if bead1Index != bead2Index:
                        bondBeadsTmp.append((bead1Index,bead2Index))
            else:
                self.log.debug(f"While generating enm, the chain {ch1Index} or the chain {ch2Index} has been found in the all atom model but not in CG")

        bondBeads = {bnd:0 for bnd in set(bondBeadsTmp)}

        for bnd in bondBeadsTmp:
            bondBeads[bnd]+=1

        return bondBeads

    def __generateNC(self,structure,cgMap,ncCut):

        atom2bead = {}
        chainsCg = set()
        #Invert map
        for bead,atomsList in cgMap.items():
            chId      = bead[1]
            chainsCg.add(chId) #Not all chains can be in the cg model

            beadIndex = bead[4]
            for atm in atomsList:
                atomIndex = atm[4]
                atom2bead[atomIndex] = beadIndex

        atomsCA      = [atm for atm in structure.get_atoms() if atm.get_name() == "CA"]
        atomsCACoord = np.asarray([atm.get_coord() for atm in structure.get_atoms() if atm.get_name() == "CA"])

        kd = cKDTree(atomsCACoord)
        ncCAAtoms = kd.query_pairs(ncCut)

        ncBeadsTmp = []
        for nc in ncCAAtoms:
            mdl1Index = atomsCA[nc[0]].get_parent().get_parent().get_parent().get_id()
            mdl2Index = atomsCA[nc[1]].get_parent().get_parent().get_parent().get_id()

            ch1Index = atomsCA[nc[0]].get_parent().get_parent().get_id()
            ch2Index = atomsCA[nc[1]].get_parent().get_parent().get_id()

            if (ch1Index in chainsCg) and (ch2Index in chainsCg):
                if ch1Index != ch2Index or mdl1Index != mdl2Index:
                    bead1Index = atom2bead[atomsCA[nc[0]].get_serial_number()]
                    bead2Index = atom2bead[atomsCA[nc[1]].get_serial_number()]
                    ncBeadsTmp.append((bead1Index,bead2Index))
            else:
                self.log.debug(f"While generating native contacts, the chain {ch1Index} or the chain {ch2Index} has been found in the all atom model but not in CG")


        ncBeads = {nc:0 for nc in set(ncBeadsTmp)}

        for nc in ncBeadsTmp:
            ncBeads[nc]+=1

        return ncBeads


    def __init__(self,name:str,inputPDBfilePath:str,aggregateChains=True,debug=False):
        self.objectName = "SBCG"
        super().__init__(name,inputPDBfilePath,True,True,aggregateChains,debug)

        self.bonds = None
        self.nativeContacts = None
        self.exclusions = None

    def generateModel(self,resolution,S,e0=0.3,eS=0.05,l0=0.2,lS=0.01,minBeads=1):

        self.log.info(f"Generating coarse grained model (SBCG) ...")

        cgAggregatedMap = {}
        self.cgMap = {}

        self.aggregatedCgStructure = Structure.Structure(self.inputStructure.get_id()+"_SBCG")

        atomCount = 1
        for mdl in self.aggregatedStructure.get_models():

            mdl_cg = Model.Model(mdl.get_id())
            self.aggregatedCgStructure.add(mdl_cg)

            for ch in mdl.get_chains():
                for clsName in self.classes.keys():

                    chName = self.classes[clsName]["leader"]
                    if ch.get_id() == chName:

                        chAtoms   = list(ch.get_atoms())

                        positions = np.asarray([atm.get_coord() for atm in chAtoms])
                        masses    = np.asarray([atm.mass for atm in chAtoms])

                    else:
                        continue

                    self.log.info(f"Working in class {clsName} which leader is {chName}.")
                    positions_cg = self.__generateSBCG_raw(positions,masses,resolution,S,e0,eS,l0,lS,minBeads)
                    Ncg = len(positions_cg)

                    ##########################
                    #Voronoi

                    if Ncg > 0:

                        ch_cg = Chain.Chain(ch.get_id())
                        mdl_cg.add(ch_cg)

                        kd = cKDTree(positions_cg)
                        allIndex2cgIndex = kd.query(positions)[1]

                        cgIndex2allAtoms = []
                        for allIndex,cgIndex in enumerate(allIndex2cgIndex):
                            while len(cgIndex2allAtoms) < cgIndex+1:
                                cgIndex2allAtoms.append([])
                            cgIndex2allAtoms[cgIndex].append(chAtoms[allIndex])

                        for cgIndex in range(Ncg):

                            atmList = cgIndex2allAtoms[cgIndex]

                            ##########################

                            chName = self.classes[clsName]["leader"]

                            cgName   = chName+str(cgIndex)
                            cgPos    = super()._CoarseGrainedBase__computeAtomListCOM(atmList)
                            cgMass   = super()._CoarseGrainedBase__computeAtomListMass(atmList)
                            cgRadius = super()._CoarseGrainedBase__computeAtomListRadiusOfGyration(atmList)
                            if(self.chargeInInput):
                                cgCharge = super()._CoarseGrainedBase__computeAtomListCharge(atmList)
                            else:
                                cgCharge = super()._CoarseGrainedBase__computeAtomListChargeFromResidues(atmList)

                            sasaPolar,sasaApolar = super()._CoarseGrainedBase__computeAtomListSASA(atmList)

                            ##########################

                            res_cg = Residue.Residue((' ',cgIndex,' '),cgName,cgIndex)
                            ch_cg.add(res_cg)

                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                atm_cg = Atom.Atom(cgName,
                                                   cgPos,
                                                   0.0,
                                                   1.0,
                                                   ' ',
                                                   cgName,
                                                   atomCount);

                                atm_cg.mass   = cgMass
                                atm_cg.radius = cgRadius
                                atm_cg.set_charge(cgCharge)

                                atm_cg.totalSASA = sasaPolar+sasaApolar
                                atm_cg.totalSASApolar  = sasaPolar
                                atm_cg.totalSASAapolar = sasaApolar

                                atm_cg.element = "X"

                                res_cg.add(atm_cg)
                                atomCount+=1

                            ##########################

                            currentBead = (mdl_cg.get_id(),ch_cg.get_id(),cgIndex,cgName)

                            cgAggregatedMap[currentBead]=[]
                            for atm in atmList:
                                mdl_id = atm.get_parent().get_parent().get_parent().get_id()
                                ch_id  = atm.get_parent().get_parent().get_id()
                                res_id = atm.get_parent().get_id()[1]
                                atm_id = atm.get_name()
                                currentAtom = (mdl_id,ch_id,res_id,atm_id)
                                cgAggregatedMap[currentBead].append(currentAtom)
                    else:
                        self.log.info(f"Class {clsName} which leader is {chName} has less beads than minBeads({minBeads}). Ignoring this chain.")

        self.spreadedCG = super()._CoarseGrainedBase__spreadStructure(self.aggregatedCgStructure,self.classes)

        ch2leader = {}
        for clsName in self.classes:
            leader = self.classes[clsName]["leader"]
            ch2leader[leader]=[leader]
            for mmb in self.classes[clsName]["members"]:
                ch2leader[mmb]=leader

        atm2index = {}
        for atm in self.spreadedStructure.get_atoms():
            mdl_id = atm.get_parent().get_parent().get_parent().get_id()
            ch_id  = atm.get_parent().get_parent().get_id()
            res_id = atm.get_parent().get_id()[1]
            atm_id = atm.get_name()
            atm2index[(mdl_id,ch_id,res_id,atm_id)] = atm.get_serial_number()

        mdl_cg = list(self.aggregatedCgStructure.get_models())[0].get_id()
        for bead in self.spreadedCG.get_atoms():
            mdl_id = bead.get_parent().get_parent().get_parent().get_id()
            ch_id  = bead.get_parent().get_parent().get_id()
            res_id = bead.get_parent().get_id()[1]
            atm_id = bead.get_name()

            currentBead = (mdl_id,ch_id,res_id,atm_id,bead.get_serial_number())

            self.cgMap[currentBead] = []
            for atm in cgAggregatedMap[(mdl_cg,ch2leader[ch_id],res_id,atm_id)]:
                mdl_atm,ch_atm,res_atm,atm_atm = atm
                index = atm2index[(mdl_id,ch_id,res_atm,atm_atm)]
                self.cgMap[currentBead].append((mdl_id,ch_id,res_atm,atm_atm,index))

        self.log.info(f"Model generation end")

    def generateTopology(self,model:dict):

        if "bondsModel" in model.keys():
            bondsModel = model["bondsModel"]
        else:
            bondsModel = None

        if "nativeContactsModel" in model.keys():
            nativeContactsModel = model["nativeContactsModel"]
        else:
            nativeContactsModel = None

        #########################################

        self.log.info(f"Generating topology ...")

        if bondsModel is not None:
            name = bondsModel["name"]
            if name == "ENM":
                self.log.info(f"Generating ENM bonds ...")
                enmCut = bondsModel["parameters"]["enmCut"]
                self.bonds = self.__generateENM(self.spreadedStructure,self.cgMap,enmCut)
            else:
                self.log.critical(f"Bonds model {name} is not availble")
                sys.exit(1)

        if nativeContactsModel["name"] is not None:
            name = nativeContactsModel["name"]
            if name == "CA":
                self.log.info(f"Generating CA native contacts ...")
                ncCut = nativeContactsModel["parameters"]["ncCut"]
                self.nativeContacts = self.__generateNC(self.spreadedStructure,self.cgMap,ncCut)
            else:
                self.log.critical(f"Native contacts model {name} is not availble")
                sys.exit(1)

        #########################################

        self.exclusions = {}

        for bead in self.spreadedCG.get_atoms():
            self.exclusions[bead.get_serial_number()]=set()

        if self.bonds:
            for bnd in self.bonds.keys():
                id_i,id_j = bnd
                self.exclusions[id_i].add(id_j)
                self.exclusions[id_j].add(id_i)

        if self.nativeContacts:
            for nc in self.nativeContacts.keys():
                id_i,id_j = nc
                self.exclusions[id_i].add(id_j)
                self.exclusions[id_j].add(id_i)


        self.log.info(f"Topology generation end")

    def generateSimulation(self,filename,K,epsilon,D):

        self.log.info(f"Generating simulation ...")

        beads = [b for b in self.spreadedCG.get_atoms()]

        simulation = {}
        simulation["topology"] = {}
        simulation["topology"]["particleTypes"] = {}
        simulation["topology"]["forceField"]    = {}

        #Coordinates

        simulation["coordinates"] = {"labels":["id","position"],"data":[]}

        for atm in self.spreadedCG.get_atoms():
            simulation["coordinates"]["data"].append([atm.get_serial_number(),list(atm.get_coord())])

        #Generate types
        types={}
        for atm in list(self.spreadedCG.get_models())[0].get_atoms():
            typeName = atm.get_name()

            if typeName not in types.keys():

                mass      = round(atm.mass,3)
                radius    = round(atm.radius,3)
                charge    = round(atm.get_charge(),3)
                totalSASA = round(atm.totalSASA,3)
                totalSASApolar  = round(atm.totalSASApolar,3)
                totalSASAapolar = round(atm.totalSASAapolar,3)

                types[typeName]={"name":typeName,"mass":mass,"radius":radius,"charge":charge}

        simulation["topology"]["particleTypes"]["labels"] = ["name", "mass", "radius", "charge"]
        simulation["topology"]["particleTypes"]["data"]   = []

        for t in types.keys():
            name   = types[t]["name"]
            mass   = types[t]["mass"]
            radius = types[t]["radius"]
            charge = types[t]["charge"]
            simulation["topology"]["particleTypes"]["data"].append([name,mass,radius,charge])

        ######################################

        simulation["topology"]["structure"] = {}
        simulation["topology"]["structure"]["labels"] = ["id", "type", "modelId"]
        simulation["topology"]["structure"]["data"]   = []

        for atm in self.spreadedCG.get_atoms():
            #mdl = atm.get_parent().get_parent().get_parent().get_id()
            mdl = 0
            simulation["topology"]["structure"]["data"].append([atm.get_serial_number(),atm.get_name(),0])

        ######################################

        if self.bonds:
            simulation["topology"]["forceField"]["bonds"] = {}
            simulation["topology"]["forceField"]["bonds"]["type"]       = ["Bond2","HarmonicConst_K"]
            simulation["topology"]["forceField"]["bonds"]["parameters"] = {"K":K}
            simulation["topology"]["forceField"]["bonds"]["labels"]     = ["id_i", "id_j", "r0"]
            simulation["topology"]["forceField"]["bonds"]["data"]       = []

            for bnd in self.bonds.keys():
                id_i,id_j = bnd
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                dst = round(np.linalg.norm(pos_i-pos_j),3)
                simulation["topology"]["forceField"]["bonds"]["data"].append([id_i,id_j,dst])


        ######################################

        if self.nativeContacts:
            simulation["topology"]["forceField"]["nativeContacts"] = {}
            simulation["topology"]["forceField"]["nativeContacts"]["type"]       = ["Bond2","MorseWCA"]
            simulation["topology"]["forceField"]["nativeContacts"]["parameters"] = {"eps0":1.0}
            simulation["topology"]["forceField"]["nativeContacts"]["labels"]     = ["id_i", "id_j", "r0", "E","D"]
            simulation["topology"]["forceField"]["nativeContacts"]["data"]       = []

            for nc in self.nativeContacts.keys():
                id_i,id_j = nc
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                dst = round(np.linalg.norm(pos_i-pos_j),3)
                E   = epsilon*self.nativeContacts[nc]
                simulation["topology"]["forceField"]["nativeContacts"]["data"].append([id_i,id_j,dst,E,D])

        ######################################

        simulation["topology"]["forceField"]["exclusions"] = {}
        simulation["topology"]["forceField"]["exclusions"]["type"]       = ["Exclusions", "ExclusionsList"]
        simulation["topology"]["forceField"]["exclusions"]["labels"]     = ["id", "id_list"]
        simulation["topology"]["forceField"]["exclusions"]["data"]       = []

        for excl in self.exclusions.keys():
            simulation["topology"]["forceField"]["exclusions"]["data"].append([excl,sorted(self.exclusions[excl])])

        ######################################

        simulation["topology"]["forceField"]["verletList"] = {}
        simulation["topology"]["forceField"]["verletList"]["type"]       = ["NeighbourList", "ConditionedVerletList"]
        simulation["topology"]["forceField"]["verletList"]["parameters"] = {"cutOffVerletFactor": 1.5,
                                                                            "exclusions": ["topology", "forceField", "exclusions"]}

        ######################################

        simulation["topology"]["forceField"]["steric"] = {}
        simulation["topology"]["forceField"]["steric"]["type"]       = ["NonBonded", "WCAType2"]
        simulation["topology"]["forceField"]["steric"]["parameters"] = {"cutOffFactor": 2.5,"condition":"intra"}
        simulation["topology"]["forceField"]["steric"]["labels"]     = ["name_i","name_j","epsilon","sigma"]
        simulation["topology"]["forceField"]["steric"]["data"]       = []

        for t1,t2 in itertools.product(types.keys(),repeat=2):
            r1 = types[t1]["radius"]
            r2 = types[t2]["radius"]

            simulation["topology"]["forceField"]["steric"]["data"].append([t1,t2,1.0,round(r1+r2,3)])

        ######################################

        self.log.info(f"Writing simulation ...")

        with open(filename,"w") as f:
            opts = jsbeautifier.default_options()
            opts.indent_size  = 2
            f.write(jsbeautifier.beautify(json.dumps(simulation), opts))
