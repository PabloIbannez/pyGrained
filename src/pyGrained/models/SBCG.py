from .. import CoarseGrainedBase

import warnings

import itertools

import json
import jsbeautifier

from tqdm import tqdm

import numpy as np
import random

from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

from scipy.spatial import cKDTree

from Bio.PDB import *

from ..utils.computeK import computeK
from ..utils.atomList import *

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
                self.logger.debug(f"While generating native contacts, the chain {ch1Index} or the chain {ch2Index} has been found in the all atom model but not in CG")


        ncBeads = {nc:0 for nc in set(ncBeadsTmp)}

        for nc in ncBeadsTmp:
            ncBeads[nc]+=1

        return ncBeads

    def __init__(self,
                 name:str,
                 inputPDBfilePath:str,
                 params:dict,
                 debug = False):

        super().__init__(tpy  = "SBCG",
                         name = name,
                         inputPDBfilePath = inputPDBfilePath,
                         removeHydrogens = True,removeNucleics  = True,
                         centerInput = params.get("centerInput",True),
                         SASA = params.get("SASA",True),
                         aggregateChains = params.get("aggregateChains",True),
                         debug = debug)

        #We have to set types,states,structure and forceField

        #
        self.bonds = None
        self.nativeContacts = None
        self.exclusions = None

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

                            chName = self.classes[clsName]["leader"]

                            cgName   = chName+str(cgIndex)
                            cgPos    = computeAtomListCOM(atmList)
                            cgMass   = computeAtomListMass(atmList)
                            cgRadius = computeAtomListRadiusOfGyration(atmList)
                            if(self.chargeInInput):
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

                            cgAggregatedMap[currentBead]=[]
                            for atm in atmList:
                                mdl_id = atm.get_parent().get_parent().get_parent().get_id()
                                ch_id  = atm.get_parent().get_parent().get_id()
                                res_id = atm.get_parent().get_id()[1]
                                atm_id = atm.get_name()
                                currentAtom = (mdl_id,ch_id,res_id,atm_id)
                                cgAggregatedMap[currentBead].append(currentAtom)
                    else:
                        self.logger.info(f"Class {clsName} which leader is {chName} has less beads than minBeads({minBeads}). Ignoring this chain.")

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

        self.logger.info(f"Model generation end")


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

        self.logger.info(f"Generating topology ...")

        if bondsModel is not None:
            name = bondsModel["name"]
            if name == "ENM":
                self.logger.info(f"Generating ENM bonds ...")
                enmCut = bondsModel["parameters"]["enmCut"]
                self.bonds = self.__generateENM(self.spreadedStructure,self.cgMap,enmCut)
            else:
                self.logger.critical(f"Bonds model {name} is not availble")
                sys.exit(1)

        if nativeContactsModel["name"] is not None:
            name = nativeContactsModel["name"]
            if name == "CA":
                self.logger.info(f"Generating CA native contacts ...")
                ncCut = nativeContactsModel["parameters"]["ncCut"]
                self.nativeContacts = self.__generateNC(self.spreadedStructure,self.cgMap,ncCut)
            else:
                self.logger.critical(f"Native contacts model {name} is not availble")
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


        self.logger.info(f"Topology generation end")

    def generateSimulation(self,filename,K,epsilon,D):

        self.logger.info(f"Generating simulation ...")

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
                if SASA:
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

        self.logger.info(f"Writing simulation ...")

        with open(filename,"w") as f:
            opts = jsbeautifier.default_options()
            opts.indent_size  = 2
            f.write(jsbeautifier.beautify(json.dumps(simulation), opts))
