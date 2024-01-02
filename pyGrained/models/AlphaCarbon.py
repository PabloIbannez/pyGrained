from .. import CoarseGrainedBase

import warnings

import itertools

import json

from tqdm import tqdm

import numpy as np

from scipy.spatial import cKDTree

from Bio.PDB import *

from ..utils.atomList import *

from ..utils.geometric import *

from ..utils.coarseGrained import *

from ..utils.bonds import *
from ..utils.nativeContacts import *

class AlphaCarbon(CoarseGrainedBase):

    def __init__(self,
                 name:str,
                 inputPDBfilePath:str,
                 fixPDB,
                 params:dict,
                 debug = False):

        super().__init__(tpy  = "AlphaCarbon",
                         name = name,
                         inputPDBfilePath = inputPDBfilePath,
                         fixPDB = fixPDB,
                         removeHetatm = True, removeHydrogens = False,removeNucleics  = True,
                         centerInput = params.get("centerInput",True),
                         SASA = params.get("SASA",False),
                         aggregateChains = params.get("aggregateChains",True),
                         debug = debug)

        #We have to set types,states,structure and forceField

        #####################################################
        ################### GENERATE MODEL ##################

        aggregatedCgMap = {}
        spreadedCgMap   = {}

        aggregatedCgStructure = Structure.Structure(self.getInputStructure().get_id()+"_AlphaCarbon")

        atomCount = 1
        for mdl in self.getAggregatedStructure().get_models():

            mdl_cg = Model.Model(mdl.get_id())
            aggregatedCgStructure.add(mdl_cg)

            for ch in mdl.get_chains():
                for clsName in self.getClasses().keys():

                    chName = self.getClasses()[clsName]["leader"]
                    if ch.get_id() == chName:
                        chAtoms = list(ch.get_atoms())
                    else:
                        continue

                    self.logger.info(f"Working in class {clsName} which leader is {chName}.")

                    ch_cg = Chain.Chain(ch.get_id())
                    mdl_cg.add(ch_cg)

                    for atm in chAtoms:
                        if atm.get_name() == "CA":

                            atmList = [atm] #Only one atom in the atom list

                            cgName   = atm.get_parent().get_resname()
                            cgIndex  = atomCount
                            cgMass   = computeAtomListMassFromResidues(atmList)
                            cgRadius = computeAtomListRadiusFromResidues(atmList)
                            cgCharge = computeAtomListChargeFromResidues(atmList)

                            cgPos    = atm.get_coord()

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

                                atm_cg.mass    = cgMass
                                atm_cg.radius  = cgRadius
                                atm_cg.set_charge(cgCharge)
                                atm_cg.element = "X"

                                res_cg.add(atm_cg)
                                atomCount+=1

                            currentBead = (mdl_cg.get_id(),ch_cg.get_id(),cgIndex,cgName)

                            aggregatedCgMap[currentBead]=[]
                            for atm in atmList:
                                mdl_id = atm.get_parent().get_parent().get_parent().get_id()
                                ch_id  = atm.get_parent().get_parent().get_id()
                                res_id = atm.get_parent().get_id()[1]
                                atm_id = atm.get_name()
                                currentAtom = (mdl_id,ch_id,res_id,atm_id)
                                aggregatedCgMap[currentBead].append(currentAtom)


        spreadedCgStructure = super()._CoarseGrainedBase__spreadStructure(aggregatedCgStructure,self.getClasses())

        spreadedCgMap = generateSpreadedCgMap(self.getSpreadedStructure(),
                                              self.getClasses(),
                                              aggregatedCgStructure,
                                              spreadedCgStructure,
                                              aggregatedCgMap)

        self.setAggregatedCgStructure(aggregatedCgStructure)
        self.setSpreadedCgStructure(spreadedCgStructure)
        self.setAggregatedCgMap(aggregatedCgMap)
        self.setSpreadedCgMap(spreadedCgMap)

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

        self.setTypes(generateTypes(spreadedCgStructure))
        self.setState(generateState(spreadedCgStructure))
        self.setStructure(generateStructure(spreadedCgStructure))

class ElasticNetworkModel(AlphaCarbon):

    def __generateENM(self,structure,enmCut):

        beads      = [b for b in structure.get_atoms()]
        beadsCoord = np.asarray([b.get_coord() for b in structure.get_atoms()])

        kd    = cKDTree(beadsCoord)
        pairs = kd.query_pairs(enmCut)

        bonds = []
        for p in pairs:
            index_i = beads[p[0]].get_serial_number()
            index_j = beads[p[1]].get_serial_number()
            if index_i < index_j:
                bonds.append((index_i,index_j))
            else:
                bonds.append((index_j,index_i))

        #Sort
        bonds = sorted(bonds,key=lambda x: x[0])

        if len(list(set(bonds))) != len(bonds):
            self.logger.error("There are repeated bonds in the ENM.")
            raise Exception("Error in the ENM generation.")

        return bonds

    def __init__(self,
                 name:str,
                 inputPDBfilePath:str,
                 params:dict,
                 debug = False):

        fixPDB = params.get("fixPDB",False)

        super().__init__(name   = name,
                         inputPDBfilePath = inputPDBfilePath,
                         fixPDB = fixPDB,
                         params = params,
                         debug  = debug)

        self.logger.info(f"Creating ENM object {name} from {inputPDBfilePath}")

        modelParams = params.get("parameters",{})

        K      = modelParams.get("K",20.0)
        enmCut = modelParams.get("enmCut",10.0)

        self.logger.info(f"K: {K}")
        self.logger.info(f"enmCut: {enmCut}")

        ######################### TOPOLOGY ##########################

        cgMap = self.getSpreadedCgMap()

        bonds = self.__generateENM(self.getSpreadedCgStructure(),enmCut)

        ####################### FORCE FIELD #########################

        #ForceField

        self.logger.info(f"Generating force field ...")

        beads = [b for b in self.getSpreadedCgStructure().get_atoms()]

        forceField = {}

        forceField["bonds"] = {}
        forceField["bonds"]["type"]       = ["Bond2","HarmonicCommon_K"]
        forceField["bonds"]["parameters"] = {"K":K}
        forceField["bonds"]["labels"] = ["id_i", "id_j", "r0"]
        forceField["bonds"]["data"]   = []

        for bnd in bonds:
            id_i,id_j = bnd
            pos_i = beads[id_i].get_coord()
            pos_j = beads[id_j].get_coord()
            r0 = np.linalg.norm(pos_i-pos_j)
            forceField["bonds"]["data"].append([id_i,id_j,r0])

        #self.logger.debug(f"Force field: {forceField}")
        self.logger.info(f"Force field generation end")

        #ForceField end

        self.setForceField(forceField)

class SelfOrganizedPolymer(AlphaCarbon):

    def __init__(self,
                 name:str,
                 inputPDBfilePath:str,
                 params:dict,
                 debug = False):

        fixPDB = params.get("fixPDB",False)

        super().__init__(name   = name,
                         inputPDBfilePath = inputPDBfilePath,
                         fixPDB = fixPDB,
                         params = params,
                         debug  = debug)

        self.logger.info(f"Creating SelfOrganizedPolymer object {name} from {inputPDBfilePath}")

        modelParams = params.get("parameters",{})

        epsNC  = modelParams.get("epsilonNC",1.0)
        self.logger.info(f"epsilonNC: {epsNC}")

        ######################### TOPOLOGY ##########################

        cgMap = self.getSpreadedCgMap()

        bonds          = generateBondsFromStructure(self.getSpreadedCgStructure())
        nativeContacts = generateNativeContactsAlphaCarbonByCutOffFromStructure(self.getSpreadedStructure(),8.0,2)

        invCgMap = generateInverseIndexMap(cgMap)
        nativeContacts = [(invCgMap[atom1],invCgMap[atom2]) for atom1,atom2 in nativeContacts.keys()]

        exclusions = {}

        for bead in self.getSpreadedCgStructure().get_atoms():
            exclusions[bead.get_serial_number()]=set()

        for bnd in bonds:
            id_i,id_j = bnd
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for nc in nativeContacts:
            id_i,id_j = nc
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        ####################### FORCE FIELD #########################

        #ForceField

        self.logger.info(f"Generating force field ...")

        beads = [b for b in self.getSpreadedCgStructure().get_atoms()]

        forceField = {}

        forceField["bonds"] = {}
        forceField["bonds"]["type"]       = ["Bond2","FeneCommon_K_R0"]
        forceField["bonds"]["parameters"] = {"K":20.15,"R0":2.0} #14 N/m = 20.15 Kcal/mol/A^2
        forceField["bonds"]["labels"] = ["id_i", "id_j", "r0"]
        forceField["bonds"]["data"]   = []

        for bnd in bonds:
            id_i,id_j = bnd
            pos_i = beads[id_i].get_coord()
            pos_j = beads[id_j].get_coord()
            r0 = np.linalg.norm(pos_i-pos_j)
            forceField["bonds"]["data"].append([id_i,id_j,r0])

        forceField["nativeContacts"] = {}
        forceField["nativeContacts"]["type"]       = ["Bond2","LennardJonesType2Common_epsilon"]
        forceField["nativeContacts"]["parameters"] = {"epsilon":epsNC}
        forceField["nativeContacts"]["labels"]     = ["id_i", "id_j", "sigma"]
        forceField["nativeContacts"]["data"]       = []

        for nc in nativeContacts:
            id_i,id_j = nc
            pos_i = beads[id_i].get_coord()
            pos_j = beads[id_j].get_coord()
            dst = round(np.linalg.norm(pos_i-pos_j),3)
            forceField["nativeContacts"]["data"].append([id_i,id_j,dst])

        #Verlet list

        forceField["nl"] = {}
        forceField["nl"]["type"]       = ["VerletConditionalListSet","nonExclIntra_nonExclInter"]
        forceField["nl"]["parameters"] = {"cutOffVerletFactor":1.3}
        forceField["nl"]["labels"]     = ["id", "id_list"]
        forceField["nl"]["data"]       = []

        for bead in self.getSpreadedCgStructure().get_atoms():
            id_ = bead.get_serial_number()
            forceField["nl"]["data"].append([id_,list(exclusions[id_])])

        #Steric

        forceField["steric"] = {}
        forceField["steric"]["type"]       = ["NonBonded", "Steric6"]
        forceField["steric"]["parameters"] = {"cutOffFactor": 2.5,"condition":"intra"}
        forceField["steric"]["labels"]     = ["name_i","name_j","epsilon","sigma"]
        forceField["steric"]["data"]       = []


        types     = self.getTypes()
        typeNames = [types[t]["name"] for t in types.keys()]
        for tName1,tName2 in itertools.product(typeNames,repeat=2):
            forceField["steric"]["data"].append([tName1,tName2,1.0,3.8])

        #self.logger.debug(f"Force field: {forceField}")
        self.logger.info(f"Force field generation end")

        #ForceField end

        self.setForceField(forceField)

class KaranicolasBrooks(AlphaCarbon):

    def __init__(self,
                 name:str,
                 inputPDBfilePath:str,
                 params:dict,
                 debug = False):

        fixPDB = params.get("fixPDB",True) # We need hydrogens to be present in the PDB file

        super().__init__(name   = name,
                         inputPDBfilePath = inputPDBfilePath,
                         fixPDB = fixPDB,
                         params = params,
                         debug  = debug)


        Tf = 350
        epsRes = Tf*0.0054

        cutOff = 25.0
        nexcl  = 4

        ######################### TOPOLOGY ##########################

        cgMap = self.getSpreadedCgMap()

        bonds         = generateBondsFromStructure(self.getSpreadedCgStructure())
        angles        = generateAnglesFromBonds(bonds)
        dihedrals     = generateDihedralsFromBonds(bonds)

        nativeContacts = generateKaranicolasBrooksNativeContactsFromStructure(self.getSpreadedStructure(),cutOff,nexcl)

        #From all atom to coarsegrained
        invCgMap = generateInverseIndexMap(cgMap)
        nativeContacts = {(invCgMap[atom1],invCgMap[atom2]): nativeContacts[(atom1,atom2)] for atom1,atom2 in nativeContacts.keys()}

        #############################################################

        #Compute inner radius of the beads
        beads       = list(self.getSpreadedCgStructure().get_atoms())
        positions   = np.array([bead.get_coord() for bead in beads])
        mdls        = [b.get_parent().get_parent().get_parent().get_id() for b in beads]
        innerRadius = [float("inf") for b in beads]

        tree  = cKDTree(positions)
        pairs = tree.query_pairs(cutOff)

        for pair in pairs:
            i,j = pair

            res_seq_i = beads[i].get_parent().get_id()[1]
            res_seq_j = beads[j].get_parent().get_id()[1]

            ch_i = beads[i].get_parent().get_parent().get_id()
            ch_j = beads[j].get_parent().get_parent().get_id()

            #Check if (i,j) is a native contact
            if (i,j) in nativeContacts or (j,i) in nativeContacts:
                continue

            if ch_i == ch_j:
                if abs(res_seq_i-res_seq_j) > nexcl:
                    r = np.linalg.norm(positions[i]-positions[j])
                    innerRadius[i] = min(innerRadius[i],r)
                    innerRadius[j] = min(innerRadius[j],r)
            else:
                r = np.linalg.norm(positions[i]-positions[j])
                innerRadius[i] = min(innerRadius[i],r)
                innerRadius[j] = min(innerRadius[j],r)

        for r in innerRadius:
            r/=2.0
            r*=np.power(2.0,1.0/6.0)

        exclusions = {}

        for bead in self.getSpreadedCgStructure().get_atoms():
            exclusions[bead.get_serial_number()]=set()

        for bnd in bonds:
            id_i,id_j = bnd
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for ang in angles:
            id_i,id_j,id_k = ang
            exclusions[id_i].add(id_j)
            exclusions[id_i].add(id_k)

            exclusions[id_j].add(id_i)
            exclusions[id_j].add(id_k)

            exclusions[id_k].add(id_i)
            exclusions[id_k].add(id_j)

        for nc in nativeContacts:
            id_i,id_j = nc
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        if True:
            totalNCenergy = 0.0
            for nc in nativeContacts.keys():
                totalNCenergy += nativeContacts[nc]["energy"]

            epsNC = epsRes*len(beads)/totalNCenergy

            for nc in nativeContacts:
                nativeContacts[nc]["energy"] *= epsNC
        else:
            pass

        ########################## STATE ############################

        state = self.getState()

        state["labels"].append("innerRadius")

        for index in range(len(state["data"])):
            id_ = state["data"][index][0]
            state["data"][index].append(innerRadius[id_])

        self.setState(state)

        ######################## FORCE FIELD ########################

        #ForceField

        self.logger.info(f"Generating force field ...")

        beads = [b for b in self.getSpreadedCgStructure().get_atoms()]

        forceField = {}

        forceField["bonds"] = {}
        forceField["bonds"]["type"]       = ["Bond2","HarmonicCommon_K"]
        forceField["bonds"]["parameters"] = {"K":2.0*200*epsRes}
        forceField["bonds"]["labels"]     = ["id_i", "id_j", "r0"]
        forceField["bonds"]["data"]       = []

        for bnd in bonds:
            id_i,id_j = bnd
            pos_i = beads[id_i].get_coord()
            pos_j = beads[id_j].get_coord()
            r0 = np.linalg.norm(pos_i-pos_j)
            forceField["bonds"]["data"].append([id_i,id_j,r0])


        forceField["angles"] = {}
        forceField["angles"]["type"]       = ["Bond3","HarmonicAngularCommon_K"]
        forceField["angles"]["parameters"] = {"K":2.0*40*epsRes}
        forceField["angles"]["labels"]     = ["id_i", "id_j", "id_k", "theta0"]
        forceField["angles"]["data"]       = []

        for ang in angles:
            id_i,id_j,id_k = ang
            theta0 = atomAngle(beads[id_i],beads[id_j],beads[id_k])
            forceField["angles"]["data"].append([id_i,id_j,id_k,theta0])

        forceField["dihedrals"] = {}
        forceField["dihedrals"]["type"]       = ["Bond4","Dihedral4"]
        forceField["dihedrals"]["parameters"] = {}
        forceField["dihedrals"]["labels"]     = ["id_i", "id_j", "id_k", "id_l", "phi0", "K"]
        forceField["dihedrals"]["data"]       = []

        currentPath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(currentPath,"..","data","KaranicolasBrooksDiheParm.json"),"r") as fd:
            kbDihe = json.load(fd)

        kbDiheTmp = {}
        for kbDiheInfo in kbDihe.keys():
            t1   = kbDiheInfo.split("_")[0]
            t2   = kbDiheInfo.split("_")[1]
            nDih = kbDiheInfo.split("_")[2]

            kbDiheTmp[f"{t2}_{t1}_{nDih}"] = kbDihe[kbDiheInfo]

        kbDihe.update(kbDiheTmp)

        for dih in dihedrals:
            id_i,id_j,id_k,id_l = dih

            tj = beads[id_j].get_name()
            tk = beads[id_k].get_name()

            K    = []
            phi0 = []

            for nDih in range(1,5):
                diheInfo = kbDihe[f"{tj}_{tk}_{nDih}"]
                K.append(diheInfo[0])
                phi0.append(diheInfo[1])

            forceField["dihedrals"]["data"].append([id_i,id_j,id_k,id_l,phi0,K])

        forceField["nativeContacts"] = {}
        forceField["nativeContacts"]["type"]       = ["Bond2","LennardJonesKaranicolasBrooks"]
        forceField["nativeContacts"]["parameters"] = {}
        forceField["nativeContacts"]["labels"]     = ["id_i", "id_j", "sigma", "epsilon"]
        forceField["nativeContacts"]["data"]       = []

        for nc in nativeContacts:
            id_i,id_j = nc
            pos_i = beads[id_i].get_coord()
            pos_j = beads[id_j].get_coord()
            dst = round(np.linalg.norm(pos_i-pos_j),3)
            forceField["nativeContacts"]["data"].append([id_i,id_j,dst,epsNC])

        #Verlet list

        forceField["nl"] = {}
        forceField["nl"]["type"]       = ["VerletConditionalListSet","nonExclIntra_nonExclInter"]
        forceField["nl"]["parameters"] = {"cutOffVerletFactor":1.3}
        forceField["nl"]["labels"]     = ["id", "id_list"]
        forceField["nl"]["data"]       = []

        exclusions = {}

        for bead in self.getSpreadedCgStructure().get_atoms():
            exclusions[bead.get_serial_number()]=set()

        for bnd in bonds:
            id_i,id_j = bnd
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for nc in nativeContacts:
            id_i,id_j = nc
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for bead in self.getSpreadedCgStructure().get_atoms():
            id_ = bead.get_serial_number()
            forceField["nl"]["data"].append([id_,list(exclusions[id_])])

        #Steric

        forceField["steric"] = {}
        forceField["steric"]["type"]       = ["NonBonded", "StericInner12"]
        forceField["steric"]["parameters"] = {"cutOffFactor": 2.5,
                                              "condition":"intra",
                                              "epsilon":1.5e-5*epsRes/21.5}

        #self.logger.debug(f"Force field: {forceField}")
        self.logger.info(f"Force field generation end")

        #ForceField end

        self.setForceField(forceField)
