from .. import CoarseGrainedBase

import warnings

import itertools

from tqdm import tqdm

import numpy as np
import random

from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

from scipy.spatial import cKDTree

from Bio.PDB import *

from ..utils.computeK import computeK
from ..utils.atomList import *

from ..utils.coarseGrained import *

from ..utils.bonds import *
from ..utils.nativeContacts import *

class SBCG(CoarseGrainedBase):

    def __generateSBCG_raw(self,positions,weights,resolution,S,e0,eS,l0,lS,minBeads):

        Nall   = positions.shape[0]
        Nbeads = int(Nall/resolution)+1

        self.logger.info(f"Generaiting SBCG, from {Nall} atoms to {Nbeads} beads")

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
                self.logger.debug(f"While generating enm, the chain {ch1Index} or the chain {ch2Index} has been found in the all atom model but not in CG")

        bondBeads = {bnd:0 for bnd in set(bondBeadsTmp)}

        for bnd in bondBeadsTmp:
            bondBeads[bnd]+=1

        return bondBeads

    def __generateCountBonds(self,structure,cgMap):

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
        bondCAAtoms = kd.query_pairs(5.0)

        bondBeadsTmp = []
        for bnd in bondCAAtoms:

            mdl1Index = atomsCA[bnd[0]].get_parent().get_parent().get_parent().get_id()
            mdl2Index = atomsCA[bnd[1]].get_parent().get_parent().get_parent().get_id()

            ch1Index = atomsCA[bnd[0]].get_parent().get_parent().get_id()
            ch2Index = atomsCA[bnd[1]].get_parent().get_parent().get_id()

            res1Index = atomsCA[bnd[0]].get_parent().get_id()[1]
            res2Index = atomsCA[bnd[1]].get_parent().get_id()[1]

            if (ch1Index in chainsCg) and (ch2Index in chainsCg):
                if ch1Index == ch2Index and mdl1Index == mdl2Index:
                    if abs(res1Index-res2Index) == 1:
                        bead1Index = atom2bead[atomsCA[bnd[0]].get_serial_number()]
                        bead2Index = atom2bead[atomsCA[bnd[1]].get_serial_number()]
                        if bead1Index != bead2Index:
                            bondBeadsTmp.append((bead1Index,bead2Index))
            else:
                self.logger.debug(f"While generating enm, the chain {ch1Index} or the chain {ch2Index} has been found in the all atom model but not in CG")

        bondBeads = {bnd:0 for bnd in set(bondBeadsTmp)}

        for bnd in bondBeadsTmp:
            bondBeads[bnd]+=1

        return bondBeads

    def __generateNC(self,structure,cgMap,ncCut,n):

        atom2bead = {}
        chainsCg = set()
        #Invert map
        for bead,atomsList in cgMap.items():
            chId      = bead[1]
            chainsCg.add(chId) #Not all chains could be present in the cg model

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

            res1Index = atomsCA[nc[0]].get_parent().get_id()[1]
            res2Index = atomsCA[nc[1]].get_parent().get_id()[1]

            differentChain = (ch1Index != ch2Index or mdl1Index != mdl2Index)

            if (ch1Index in chainsCg) and (ch2Index in chainsCg):
                if abs(res1Index-res2Index) > n or differentChain:
                    bead1Index = atom2bead[atomsCA[nc[0]].get_serial_number()]
                    bead2Index = atom2bead[atomsCA[nc[1]].get_serial_number()]
                    if bead1Index != bead2Index:
                        ncBeadsTmp.append((bead1Index,bead2Index))
            else:
                self.logger.debug(f"While generating native contacts, the chain {ch1Index} or the chain {ch2Index} has been found in the all atom model but not in CG")


        ncBeads = {nc:0 for nc in set(ncBeadsTmp)}

        for nc in ncBeadsTmp:
            ncBeads[nc]+=1

        self.logger.info(f"Maximum number of native contacts: {max(ncBeads.values())}")

        return ncBeads

    def __init__(self,
                 name:str,
                 inputPDBfilePath:str,
                 params:dict,
                 debug = False):

        fixPDB = params.get("fixPDB",False)
        SASA = params.get("SASA",True)

        super().__init__(tpy  = "SBCG",
                         name = name,
                         inputPDBfilePath = inputPDBfilePath,
                         fixPDB = fixPDB,
                         removeHetatm = True, removeHydrogens = False,removeNucleics  = True,
                         centerInput = params.get("centerInput",True),
                         SASA = SASA,
                         aggregateChains = params.get("aggregateChains",True),
                         debug = debug)

        #We have to set types,states,structure and forceField

        #####################################################
        ################### GENERATE MODEL ##################

        self.logger.info(f"Generating coarse grained model (SBCG) ...")

        globalParams = params["parameters"]

        resolution = globalParams["resolution"]
        S          = globalParams["steps"]

        e0         = params.get("e0",0.3)
        eS         = params.get("eS",0.05)
        l0         = params.get("l0",0.2)
        lS         = params.get("lS",0.01)
        minBeads   = params.get("minBeads",1)

        aggregatedCgMap = {}
        spreadedCgMap   = {}

        aggregatedCgStructure = Structure.Structure(self.getInputStructure().get_id()+"_SBCG")

        atomCount = 1
        for mdl in self.getAggregatedStructure().get_models():

            mdl_cg = Model.Model(mdl.get_id())
            aggregatedCgStructure.add(mdl_cg)

            for ch in mdl.get_chains():
                for clsName in self.getClasses().keys():

                    chName = self.getClasses()[clsName]["leader"]
                    if ch.get_id() == chName:

                        chAtoms   = list(ch.get_atoms())

                        positions = np.asarray([atm.get_coord() for atm in chAtoms])
                        masses    = np.asarray([atm.mass for atm in chAtoms])

                    else:
                        continue

                    self.logger.info(f"Working in class {clsName} which leader is {chName}.")
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

                            chName = self.getClasses()[clsName]["leader"]

                            cgName   = chName+str(cgIndex)
                            cgPos    = computeAtomListCOM(atmList)
                            cgMass   = computeAtomListMass(atmList)
                            cgRadius = computeAtomListRadiusOfGyration(atmList)
                            if(self.getChargeInInput()):
                                cgCharge = computeAtomListCharge(atmList)
                            else:
                                cgCharge = computeAtomListChargeFromResidues(atmList)

                            if SASA:
                                sasaPolar,sasaApolar = computeAtomListSASA(atmList)

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

                                if SASA:
                                    atm_cg.totalSASA = sasaPolar+sasaApolar
                                    atm_cg.totalSASApolar  = sasaPolar
                                    atm_cg.totalSASAapolar = sasaApolar

                                atm_cg.element = "X"

                                res_cg.add(atm_cg)
                                atomCount+=1

                            ##########################

                            currentBead = (mdl_cg.get_id(),ch_cg.get_id(),cgIndex,cgName)

                            aggregatedCgMap[currentBead]=[]
                            for atm in atmList:
                                mdl_id = atm.get_parent().get_parent().get_parent().get_id()
                                ch_id  = atm.get_parent().get_parent().get_id()
                                res_id = atm.get_parent().get_id()[1]
                                atm_id = atm.get_name()
                                currentAtom = (mdl_id,ch_id,res_id,atm_id)
                                aggregatedCgMap[currentBead].append(currentAtom)
                    else:
                        self.logger.info(f"Class {clsName} which leader is {chName} has less beads than minBeads({minBeads}). Ignoring this chain.")

        spreadedCgStructure = super()._CoarseGrainedBase__spreadStructure(aggregatedCgStructure,self.getClasses())

        spreadedCgMap = generateSpreadedCgMap(self.getSpreadedStructure(),
                                              self.getClasses(),
                                              aggregatedCgStructure,
                                              spreadedCgStructure,
                                              aggregatedCgMap)

        self.logger.info(f"Model generation end")

        #############################################################

        #We have defined the following attributes:

        #aggregatedCgStructure: The coarse grained structure for class leaders

        #spreadedCgStructure: The spreaded coarse grained structure

        #aggregatedCgMap: A dictionary that maps the coarse-grained beads to the original atoms of the class leaders.
        #                 The keys are the coarse-grained beads and the values are the original atoms.
        #                 The keys are tuples of the form (model,chain,residue,atom,serial number)
        #                 and the values are tuples of the form (model,chain,residue,atom,serial number).

        #spreadedCgMap: A dictionary that maps the coarse-grained beads to the original atoms.
        #               The keys are the coarse-grained beads and the values are the original atoms.
        #               The keys are tuples of the form (model,chain,residue,atom,serial number)
        #               and the values are tuples of the form (model,chain,residue,atom,serial number).

        #############################################################

        types     = generateTypes(spreadedCgStructure,SASA)
        state     = generateState(spreadedCgStructure)
        structure = generateStructure(spreadedCgStructure)

        #############################################################

        self.logger.info(f"Generating topology ...")

        try:
            bondsModel = globalParams["bondsModel"]
        except:
            self.logger.error(f"bondsModel not defined in params")
            raise Exception("bondsModel not defined in parameters")

        try:
            nativeContactsModel = globalParams["nativeContactsModel"]
        except:
            self.logger.error("nativeContactsModel not defined in parameters")
            raise Exception("nativeContactsModel not defined in parameters")

        self.logger.debug(f"Selected bonds model: {bondsModel}")
        self.logger.debug(f"Selected native contacts model: {nativeContactsModel}")

        #############################################################

        self.logger.info(f"Generating bonds ...")

        bondsModelName = bondsModel["name"]
        if bondsModelName == "ENM":
            self.logger.info(f"Generating ENM bonds ...")
            enmCut = bondsModel["parameters"]["enmCut"]
            bonds = self.__generateENM(self.getSpreadedStructure(),spreadedCgMap,enmCut)
        elif bondsModelName == "count":
            self.logger.info(f"Generating count bonds ...")
            bonds = self.__generateCountBonds(self.getSpreadedStructure(),spreadedCgMap)
        else:
            self.logger.error(f"Bonds model {bondsModelName} is not availble")
            raise Exception(f"Bonds model not available")

        self.logger.info(f"Generating native contacts ...")

        nativeContacsModelName = nativeContactsModel["name"]
        if nativeContacsModelName == "CA" or nativeContacsModelName == "count":
            self.logger.info(f"Generating CA native contacts ...")
            if "parameters" in nativeContactsModel:
                ncCut = nativeContactsModel["parameters"].get("ncCut",8.0)
            else:
                ncCut = 8.0
            nativeContacts = self.__generateNC(self.getSpreadedStructure(),spreadedCgMap,ncCut,2)
        else:
            self.logger.error(f"Native contacts model {nativeContacsModelName} is not availble")
            raise Exception(f"Native contacts model not available")

        self.logger.info(f"Topology generation end")

        #########################################

        #ForceField

        self.logger.info(f"Generating force field ...")

        forceField = {}

        #Auxiliar list with all beads in the system
        beads = [b for b in spreadedCgStructure.get_atoms()]

        #Bonds
        if bondsModelName == "ENM":
            forceField["bonds"] = {}
            forceField["bonds"]["type"]       = ["Bond2","HarmonicCommon_K"]
            forceField["bonds"]["parameters"] = {}
            forceField["bonds"]["parameters"]["K"] = bondsModel["parameters"]["K"]
            forceField["bonds"]["labels"] = ["id_i", "id_j", "r0"]
            forceField["bonds"]["data"]   = []

            for bnd in bonds.keys():
                id_i,id_j = bnd
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                r0 = np.linalg.norm(pos_i-pos_j)
                forceField["bonds"]["data"].append([id_i,id_j,r0])
        elif bondsModelName == "count":
            forceField["bonds"] = {}
            forceField["bonds"]["type"]       = ["Bond2","r0Count"]
            forceField["bonds"]["parameters"] = {}
            forceField["bonds"]["labels"] = ["id_i", "id_j", "r0", "n"]
            forceField["bonds"]["data"]   = []

            for bnd in bonds.keys():
                id_i,id_j = bnd
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                r0 = np.linalg.norm(pos_i-pos_j)
                forceField["bonds"]["data"].append([id_i,id_j,r0,bonds[bnd]])
        else:
            self.logger.error(f"Bonds model {bondsModelName} is not availble")
            raise Exception(f"Bonds model not available")

        #Native contacts
        if nativeContacsModelName == "CA":
            forceField["nativeContacts"] = {}
            forceField["nativeContacts"]["type"]       = ["Bond2","MorseWCACommon_eps0"]
            forceField["nativeContacts"]["parameters"] = {"eps0":1.0}
            forceField["nativeContacts"]["labels"]     = ["id_i", "id_j", "r0", "E","D"]
            forceField["nativeContacts"]["data"]       = []

            for nc in nativeContacts.keys():
                id_i,id_j = nc
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                dst = round(np.linalg.norm(pos_i-pos_j),3)
                E   = nativeContactsModel["parameters"]["epsilon"]*nativeContacts[nc]
                D   = nativeContactsModel["parameters"]["D"]
                forceField["nativeContacts"]["data"].append([id_i,id_j,dst,E,D])
        elif nativeContacsModelName == "count":

            forceField["nativeContacts"] = {}
            forceField["nativeContacts"]["type"]       = ["Bond2","roCount"]
            forceField["nativeContacts"]["parameters"] = {}
            forceField["nativeContacts"]["labels"]     = ["id_i", "id_j", "r0", "n"]
            forceField["nativeContacts"]["data"]       = []

            for nc in nativeContacts.keys():
                id_i,id_j = nc
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                dst = round(np.linalg.norm(pos_i-pos_j),3)
                forceField["nativeContacts"]["data"].append([id_i,id_j,dst,nativeContacts[nc]])
        else:
            self.logger.error(f"Native contacts model {nativeContacsModelName} is not availble")
            raise Exception(f"Native contacts model not available")

        #Verlet list

        forceField["nl"] = {}
        forceField["nl"]["type"]       = ["VerletConditionalListSet","nonExclIntra_nonExclInter"]
        forceField["nl"]["parameters"] = {"cutOffVerletFactor":1.5}
        forceField["nl"]["labels"]     = ["id", "id_list"]
        forceField["nl"]["data"]       = []

        exclusions = {}

        for bead in spreadedCgStructure.get_atoms():
            exclusions[bead.get_serial_number()]=set()

        for bnd in bonds.keys():
            id_i,id_j = bnd
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for nc in nativeContacts.keys():
            id_i,id_j = nc
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for bead in spreadedCgStructure.get_atoms():
            id_ = bead.get_serial_number()
            forceField["nl"]["data"].append([id_,list(exclusions[id_])])

        #Steric

        forceField["steric"] = {}
        forceField["steric"]["type"]       = ["NonBonded", "WCAType2"]
        forceField["steric"]["parameters"] = {"cutOffFactor": 2.5,"condition":"intra"}
        forceField["steric"]["labels"]     = ["name_i","name_j","epsilon","sigma"]
        forceField["steric"]["data"]       = []

        for t1,t2 in itertools.product(types.keys(),repeat=2):
            tName1 = types[t1]["name"]
            tName2 = types[t2]["name"]

            tRadius1 = types[t1]["radius"]
            tRadius2 = types[t2]["radius"]

            forceField["steric"]["data"].append([tName1,tName2,1.0,round(tRadius1+tRadius2,3)])

        #self.logger.debug(f"Force field: {forceField}")
        self.logger.info(f"Force field generation end")

        #ForceField end

        #############################################################

        self.setAggregatedCgStructure(aggregatedCgStructure)
        self.setSpreadedCgStructure(spreadedCgStructure)
        self.setAggregatedCgMap(aggregatedCgMap)
        self.setSpreadedCgMap(spreadedCgMap)

        self.setTypes(types)
        self.setState(state)
        self.setStructure(structure)
        self.setForceField(forceField)
