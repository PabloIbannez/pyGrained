from .. import CoarseGrainedBase

import numpy as np
# from Bio.PDB import * ## This also imports SASA
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue, Atom
from sklearn.cluster import KMeans
from copy import deepcopy
import os
import warnings
import itertools

from scipy.spatial import cKDTree
from ..utils.atomList import *
from ..utils.coarseGrained import *

import logging

class ChainAdaptiveCG:
    def __init__(self, n_beads:int, 
                 coords:np.ndarray, 
                 masses:np.ndarray, 
                 R_init:np.ndarray | None=None, sigma:float=2.0):
        self.coords = coords  # (N,3)
        self.sigma = sigma
        self.masses = masses  # (N,)

        self.logger = logging.getLogger(f"pyGrained")

        if R_init is not None:
            self.R_init = R_init.copy()
            self.n_beads = R_init.shape[0]
            logging.info(f"Using provided initial bead positions for chain with {self.n_beads} beads.")
        else:
            self.n_beads = n_beads
            self.R_init = self._initialize_beads()
            ## This for checking what happens if initial beads are all placed in the same coordinates
            ## For the moment it fails, the whole thing collapses
            # self.R_init = np.tile(np.mean(self.coords, axis=0), self.n_beads).reshape(-1,3)
            # import pdb;pdb.set_trace()
        
        self.R = self.R_init.copy()
        self.R_opt = None
        self.chi = None
        self.chi_opt = None
        # self.R_opt, self.chi_opt = self.optimize()

    def _initialize_beads(self):
        ## TODO: Test if  initializing beads in the same position lead to proper CG
        """
        Initializes bead positions using KMeans.
        """
        kmeans = KMeans(n_clusters=self.n_beads, n_init=10)
        kmeans.fit(self.coords)

        # R tendrá forma (M,3): posiciones iniciales de los beads
        return kmeans.cluster_centers_.astype(float)
    
    def compute_chi(self):
        """
        Calculates χ(r_i) para each atom and bead.
        χ_iμ = Δ(r_i - R_μ) / Σ_ν Δ(r_i - R_ν)
        where Δ is a Gaussiana with deviationsigma.
        """
        diff = self.coords[:, None, :] - self.R[None, :, :]
        dist2 = np.sum(diff**2, axis=2)  # (N,M)

        # Gaussianas (Δ)
        weights = np.exp(-dist2 / (2 * self.sigma**2))
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        sum_weights[sum_weights == 0] = 1e-12  # Avoid division by zero
        chi = weights / sum_weights

        # Normalización → χ
        # chi = weights / np.sum(weights, axis=1, keepdims=True)
        # if np.any(np.isnan(chi)):
        #     import pdb;pdb.set_trace()
        return chi

    def update_R(self, chi):
        """
        Refreshes bead positions using:
        R_μ = Σ_i [m_i r_i χ_iμ] / Σ_i [m_i χ_iμ]
        """
        # weighted = self.coords[:, None, :] * (self.masses[:, None, None] * chi)
        weighted = self.coords[:, None, :] * (self.masses[:, None, None] * chi[:, :, None])
        num = np.sum(weighted, axis=0)                  # (M,3)
        den = np.sum(self.masses[:, None] * chi, axis=0)  # (M,)

        # self.R = num / den[:, None]
        return num / den[:, None]

    def optimize(self, max_iter=100, tol=1e-4, debug=False):
        """
        Iterate until convergences.
        Convergens when any bead moves more than the tolerance (tol).
        """
        # R_old = self.R_init.copy()
        for it in range(max_iter):
            R_old = self.R.copy()

            chi = self.compute_chi()
            self.R = self.update_R(chi)

            # Cálculo del desplazamiento máximo
            shift = np.max(np.linalg.norm(self.R - R_old, axis=1))

            if shift < tol:
                self.logger.info(f"Converged in {it} iterations")
                break
        if debug:
            import pdb;pdb.set_trace()
        
        self.R_opt = self.R.copy() 
        self.chi_opt = self.compute_chi()
        # print(np.abs(self.R_init - self.R_opt))
        self.logger.info(f"Finished optimization after {it+1} iterations")
        return self.R, chi

class AdaptiveCG(CoarseGrainedBase):
    def __init__(self, name:str, 
                 inputPDBfilePath:str, 
                 params:dict, 
                 debug = False):
       
        self.inputPDBfilePath = os.path.abspath(inputPDBfilePath)
       
        globalParams = params["parameters"]
       
        self.SASA           = params.get("SASA", False)
        self.resolution     = globalParams["resolution"]
        self.sigma          = globalParams.get("sigma", 2)
        self.iterations      = globalParams.get("steps", 1000)

        self.R_0            = params.get("R_0", 20.0)
        self.minBeads       = params.get("minBeads",1)


        super().__init__(tpy  = "AdaptiveGC",
                         name = name,
                         inputPDBfilePath = inputPDBfilePath,
                         removeHetatm = True, removeHydrogens = False,removeNucleics = True,
                         centerInput = params.get("centerInput",True),
                         SASA = self.SASA,
                         aggregateChains = params.get("aggregateChains",True),
                         debug = debug)
        
        self.logger.info(f"Generating coarse grained model (AdaptiveGC) ...")
        
        # Parsing the microstate BioPython
        ## Maybe here I should be using the spreadedStructure?
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("mol", inputPDBfilePath)

        atom_coords = []
        masses = []
        micro_chains = []

        # Extract coordinates and masses 
        for atom in structure.get_atoms():
            atom_coords.append(atom.get_coord())
            masses.append(atom.mass)
            full_id = atom.get_full_id()
            micro_chains.append(full_id[2])

        self.micro_coords = np.array(atom_coords)     # (N,3)
        self.micro_masses = np.array(masses)     # (N,)
        self.micro_chains = np.array(micro_chains)     # (N,)
        # cg_chain = []
        # cg_coords = []
        # cb_beads_ids = []


        ## Iterate over each class to make initial CG
        # self.classes_beads = {}
        # self.chain_beads = {}
        # for tmp_class, chain_info in self._classes.items(): ## First calculate for the leader chain
        #     leader_chain = chain_info['leader']
        #     self.logger.info(f"Working in class {tmp_class} which leader is {leader_chain}.")
        #     tmp_coords = self.micro_coords[self.micro_chains == leader_chain]
        #     ref2orig = np.mean(tmp_coords, axis=0)
        #     n_beads = int(tmp_coords.shape[0] / self.resolution)
        #     self.logger.info(f" Chain {leader_chain} has {tmp_coords.shape[0]} atoms and will be represented with {n_beads} beads.")

        #     tmp_masses = self.micro_masses[self.micro_chains == leader_chain]
        #     tmp_chain_CG = ChainAdaptiveCG(n_beads, tmp_coords, tmp_masses, sigma=self.sigma)
        #     tmp_chain_CG.optimize(max_iter=1000)
        #     self.classes_beads[tmp_class] = deepcopy(tmp_chain_CG)
        #     self.chain_beads[leader_chain] = deepcopy(tmp_chain_CG)
        #     cg_chain.extend(leader_chain*n_beads)
        #     cb_beads_ids.extend(list(range(n_beads)))
        #     cg_coords.extend(tmp_chain_CG.R_opt)
        #     ## Now propagate to the other chains in the class
        #     # other_chains = set(chain_info["members"]) - set("P")
        #     for _, ch, trans_matrix, rot_matrix in chain_info['transformations']:
        #         if ch == leader_chain:
        #             continue
        #         self.logger.info(f" Propagating to chain {ch}.")
        #         tmp_coords_other_chain = self.micro_coords[self.micro_chains == ch]
        #         tmp_masses_other_chain = self.micro_masses[self.micro_chains == ch]
        #         beads_coords = self.classes_beads[tmp_class].R_opt.copy()
        #         R_init = (beads_coords - ref2orig) @ rot_matrix.as_matrix().T + ref2orig + trans_matrix 

        #         cg_other_chain = ChainAdaptiveCG(n_beads, tmp_coords_other_chain, tmp_masses_other_chain, sigma=self.sigma, R_init=R_init.copy())
        #         cg_other_chain.optimize(max_iter=500) 

        #         self.chain_beads[ch] = deepcopy(cg_other_chain)
                
        #         cg_chain.extend(ch*n_beads)
        #         cb_beads_ids.extend(list(range(n_beads)))
        #         cg_coords.extend(cg_other_chain.R_opt)

        # self.cg_chains = np.array(cg_chain)
        # self.cg_beads_ids = np.array(cb_beads_ids)
        # self.cg_coords = np.array(cg_coords, dtype=np.float32)

        # self.logger.info(f"Model generation end")

        # self.logger.info(f"Calculating CG distances...")

        # bead_distances = self.calculateBeadDistances(self.cg_coords, self.R_0)
        # self.bead_distances = bead_distances[0]
        # self.bead_distances_indexes = bead_distances[1]
        # self.intra_chain_distances = bead_distances[2]
        # self.inter_chain_distances = bead_distances[3]

        ## Creating a Structure class with the CG beads
        ## Using code for the SBCG class
        aggregatedCgMap = {}

        aggregatedCgStructure = Structure.Structure(self.getInputStructure().get_id()+"_AdaptiveCG")

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
                        n_beads = int(positions.shape[0] / self.resolution)

                    else:
                        continue

                    self.logger.info(f"Working in class {clsName} which leader is {chName}.")
                    ## Get the position with AdaptiveCG
                    tmp_chain_CG = ChainAdaptiveCG(n_beads, positions, masses, sigma=self.sigma)
                    tmp_chain_CG.optimize(max_iter=self.iterations)
                    
                    positions_cg = tmp_chain_CG.R_opt
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
                            ## Take care with the mass 
                            ## I have soft assignments of the atoms to the beads
                            ## So the mass should be weighted by chi
                            cgRadius = computeAtomListRadiusOfGyration(atmList)
                            if(self.getChargeInInput()):
                                cgCharge = computeAtomListCharge(atmList)
                            else:
                                cgCharge = computeAtomListChargeFromResidues(atmList)

                            if self.SASA:
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

                                if self.SASA:
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
                        self.logger.info(f"Class {clsName} which leader is {chName} has less beads than minBeads({self.minBeads}). Ignoring this chain.")

        self.spreadedCgStructure = super()._CoarseGrainedBase__spreadStructure(aggregatedCgStructure,self.getClasses())
        spreadedCgMap = generateSpreadedCgMap(self.getSpreadedStructure(),
                                              self.getClasses(),
                                              aggregatedCgStructure,
                                              self.spreadedCgStructure,
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

        types     = generateTypes(self.spreadedCgStructure,self.SASA)
        state     = generateState(self.spreadedCgStructure)
        structure = generateStructure(self.spreadedCgStructure)

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

        self.logger.info(f"Generating bonds and native contacts...")

        bondsModelName = bondsModel["name"]
        # nativeContacsModelName = nativeContactsModel["name"]
        ## TODO: add my own bonds model, with a cut off 
        if bondsModelName == "AdaptiveCG":
            self.logger.info(f"Generating AdaptiveCG bonds ...")
            adaptiveCGBondCut = bondsModel["parameters"]["adaptiveCGBondsCut"]
            adaptiveCGNativeContactCut = nativeContactsModel["parameters"]["adaptiveCGNativeContactsCut"]
            bonds, nativeContacts = self.__generateAdaptiveCGBonds(self.spreadedCgStructure,adaptiveCGBondCut,adaptiveCGNativeContactCut)
        elif bondsModelName == "ENM":
            self.logger.info(f"Generating ENM bonds ...")
            enmCut = bondsModel["parameters"]["enmCut"]
            bonds = self.__generateENM(self.getSpreadedStructure(),spreadedCgMap,enmCut)
            nativeContacts = {}
        else:
            self.logger.error(f"Bonds model {bondsModelName} is not availble")
            raise Exception(f"Bonds model not available")
        
        nativeContacsModelName = nativeContactsModel["name"]
        if nativeContacsModelName == "AdaptiveCG":
            self.info("Already generated native contacts with the AdaptiveCG bonds model.")
        elif nativeContacsModelName == "CA":
            self.logger.info(f"Generating CA native contacts ...")
            if "parameters" in nativeContactsModel:
                ncCut = nativeContactsModel["parameters"].get("ncCut",8.0)
            else:
                ncCut = 8.0
            nativeContacts = self.__generateNC(self.getSpreadedStructure(),spreadedCgMap,ncCut,2,bonds)
        else:
            self.logger.error(f"Native contacts model {nativeContacsModelName} is not availble")
            raise Exception(f"Native contacts model not available")


        self.logger.info(f"Topology generation end")
        #########################################

        #ForceField

        self.logger.info(f"Generating force field ...")

        forceField = {}

        #Auxiliar list with all beads in the system
        beads = [b for b in self.spreadedCgStructure.get_atoms()]

        #Bonds and native contacts
        if bondsModelName == "AdaptiveCG" or bondsModelName == "ENM": ##TODO: this will be the only one left
            forceField["bonds"] = {}
            forceField["bonds"]["type"]       = ["Bond2","HarmonicCommon_K"]
            forceField["bonds"]["parameters"] = {}
            forceField["bonds"]["parameters"]["K"] = bondsModel["parameters"]["K"]
            forceField["bonds"]["labels"] = ["id_i", "id_j", "r0"]
            forceField["bonds"]["data"]   = []

            for bnd in bonds:
                id_i,id_j = bnd
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                r0 = np.linalg.norm(pos_i-pos_j)
                forceField["bonds"]["data"].append([id_i,id_j,r0])
        else:
            self.logger.error(f"Bonds model {bondsModelName} is not availble")
            raise Exception(f"Bonds model not available")

        #Native contacts
        if nativeContacsModelName == "AdaptiveCG" or nativeContacsModelName == "CA": ##TODO: this will be the only one left
            forceField["nativeContacts"] = {}
            forceField["nativeContacts"]["type"]       = ["Bond2","MorseWCACommon_eps0"]
            forceField["nativeContacts"]["parameters"] = {"eps0":1.0}
            ## NOTE: removing parameters that I do not use
            # forceField["nativeContacts"]["labels"]     = ["id_i", "id_j", "r0"]
            forceField["nativeContacts"]["labels"]     = ["id_i", "id_j", "r0", "E","D"]
            forceField["nativeContacts"]["data"]       = []

            for nc in nativeContacts:
                id_i,id_j = nc
                pos_i = beads[id_i].get_coord()
                pos_j = beads[id_j].get_coord()
                dst = round(np.linalg.norm(pos_i-pos_j),3)
                E   = nativeContactsModel["parameters"]["epsilon"]*nativeContacts[nc]
                D   = nativeContactsModel["parameters"]["D"]
                forceField["nativeContacts"]["data"].append([id_i,id_j,dst,E,D])
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

        for bead in self.spreadedCgStructure.get_atoms():
            exclusions[bead.get_serial_number()]=set()

        for bnd in bonds:
            id_i,id_j = bnd
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for nc in nativeContacts:
            id_i,id_j = nc
            exclusions[id_i].add(id_j)
            exclusions[id_j].add(id_i)

        for bead in self.spreadedCgStructure.get_atoms():
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
        self.setSpreadedCgStructure(self.spreadedCgStructure)
        self.setAggregatedCgMap(aggregatedCgMap)
        self.setSpreadedCgMap(spreadedCgMap)

        self.setTypes(types)
        self.setState(state)
        self.setStructure(structure)
        self.setForceField(forceField)

    ## PHN for testing
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
        

        filteredBondBeads = {}
        for (a, b), v in bondBeads.items():
            key = tuple(sorted((a, b)))
            if key not in filteredBondBeads:
                filteredBondBeads[key] = 1

        # filteredBondBeads = {}
        # for (bead_index_1, bead_index_2), count in bondBeads.items():
        #     try:
        #         _ = filteredBondBeads[(bead_index_2, bead_index_1)]
        #         _ = filteredBondBeads[(bead_index_1, bead_index_2)]
        #         self.logger.warning(f"Bond between bead {bead_index_1} and bead {bead_index_2} already exists. Skipping.")
        #     except KeyError:
        #         filteredBondBeads[(bead_index_1, bead_index_2)] = 1
            # filteredBondBeads[(bead_index_1, bead_index_2)] = count
        # import pdb;pdb.set_trace()
        return filteredBondBeads
        # return bondBeads
    
    def __generateNC(self,structure,cgMap,ncCut,n,bondData):

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

        ## Remove duplicates and those that are already in bonds
        filteredNCBeads = {}
        for (a, b), v in ncBeads.items():
            key = tuple(sorted((a, b)))
            # import pdb;pdb.set_trace()
            if key not in filteredNCBeads and key not in bondData.keys() and (b,a) not in bondData.keys():
                filteredNCBeads[key] = 1
        return filteredNCBeads
    
    def __generateAdaptiveCGBonds(self,cgstructure,bond_cutoff, native_contacts_cutoff):
         ## Get chains
        beads = np.array([i for i in cgstructure.get_atoms()])
        bead_names = np.array([i.name for i in cgstructure.get_atoms()])
        chain_by_idx = np.array([i.get_parent().get_parent().get_id() for i in beads])
        model_by_idx = np.array([i.get_parent().get_parent().get_parent().get_id() for i in beads])
        
        ## Get coords
        coords = np.array([i.get_coord() for i in cgstructure.get_atoms()])

        kd = cKDTree(coords)

        # All pairs satisfying the cutoff
        candidate_bond_pairs = kd.query_pairs(bond_cutoff)

        bonds = {}
        
        for i, j in candidate_bond_pairs:
            bead_i_chain = chain_by_idx[i]
            bead_j_chain = chain_by_idx[j]
            bead_i_modelIdx = model_by_idx[i]
            bead_j_modelIdx = model_by_idx[j]

            # Exclude pairs of the same chain
            # if bead_i_chain != bead_j_chain:
            #     contacts[(i, j)] = 1
            # elif bead_i_chain == bead_j_chain:
            #     native_contacts[(i, j)] = 1
            ## This flipped
            ## Bonds should be within chain
            # if bead_i_modelIdx != bead_j_modelIdx or bead_i_chain != bead_j_chain:
            #     bonds[(i, j)] = 1
            # elif bead_i_modelIdx == bead_j_modelIdx and bead_i_chain == bead_j_chain:
            #     native_contacts[(i, j)] = 1

            # if bead_i_modelIdx != bead_j_modelIdx or bead_i_chain != bead_j_chain:
            #     native_contacts[(i, j)] = 1
            if bead_i_modelIdx == bead_j_modelIdx and bead_i_chain == bead_j_chain:
                bonds[(i, j)] = 1
        
        candidate_native_contact_pairs = kd.query_pairs(native_contacts_cutoff)
        native_contacts = {}

        for i, j in candidate_native_contact_pairs:
            bead_i_chain = chain_by_idx[i]
            bead_j_chain = chain_by_idx[j]
            bead_i_modelIdx = model_by_idx[i]
            bead_j_modelIdx = model_by_idx[j]

            if bead_i_modelIdx != bead_j_modelIdx or bead_i_chain != bead_j_chain:
                native_contacts[(i, j)] = 1
            # if bead_i_modelIdx == bead_j_modelIdx and bead_i_chain == bead_j_chain:
            #     bonds[(i, j)] = 1

        ## Avoid getting isolated beads in native contacts
        # import networkx as nx

        # for clsName in self.getClasses().keys():

        #     chName = self.getClasses()[clsName]["leader"]
        #     nodes = np.where(chain_by_idx == chName)[0]
         
        #     G = nx.Graph()
        #     G.add_nodes_from(nodes)
         
        #     ## Only contacts between beads in nodes
        #     for (node_a, node_b), _ in native_contacts.items():
        #         if node_a in nodes and node_b in nodes:
        #             G.add_edge(node_a, node_b)

        #         ## Find groups of connected beads in native contacts
        #     components = list(nx.connected_components(G))
        #     components_min_distances = np.eye(len(components), len(components))
        #     np.fill_diagonal(components_min_distances, np.inf)
        #     nodes_min_distance_btw_components = {}
        #     # nodes_min_distance_btw_components = np.eye(len(components), len(components))

        #     if len(components) > 1:

        #         for cc_a, cc_b in itertools.combinations(range(len(components)), 2):
                    
        #             min_dist = np.inf
        #             min_dist_pair = None
        #             cc_a_nodes = np.array(list(components[cc_a]), dtype=int)
        #             cc_b_nodes = np.array(list(components[cc_b]), dtype=int)
        #             tmp_coords_cc_a = coords[cc_a_nodes]
        #             tmp_coords_cc_b = coords[cc_b_nodes]
        #             for node_b_idx, node_b_coords in enumerate(tmp_coords_cc_b):
        #                 dist = np.linalg.norm(tmp_coords_cc_a - node_b_coords, axis=1)
        #                 tmp_min_dist = np.min(dist)
        #                 if tmp_min_dist < min_dist:
        #                     min_dist = tmp_min_dist
        #                     min_dist_pair = (int(cc_a_nodes[np.argmin(dist)]), int(cc_b_nodes[node_b_idx]))
        #             components_min_distances[cc_a, cc_b] = min_dist
        #             components_min_distances[cc_b, cc_a] = min_dist
        #             nodes_min_distance_btw_components[(cc_b, cc_a)] = min_dist_pair
        #             nodes_min_distance_btw_components[(cc_a, cc_b)] = min_dist_pair
                
        #         ## Creating the native contact
        #         for idx, dist_array in enumerate(components_min_distances[0:-1]):
        #             cc_b = np.argmin(dist_array)
        #             cc_a = idx
        #             beads_in_min_distance = nodes_min_distance_btw_components[(cc_a, cc_b)]
        #             i, j = beads_in_min_distance
        #             bead_name_i = bead_names[i]
        #             bead_name_j = bead_names[j]
        #             # import pdb;pdb.set_trace()
        #             ## Expand to the rest of models
        #             for model in np.unique(model_by_idx): ##Iterate over models
        #                 for tmp_chain in self.getClasses()[clsName].get("members"): ## Iterate over chain
        #                     tmp_bead_idx = np.where((model_by_idx == model) & (chain_by_idx == tmp_chain) & (bead_names == bead_name_i))[0][0]
        #                     tmp_bead_jdx = np.where((model_by_idx == model) & (chain_by_idx == tmp_chain) & (bead_names == bead_name_j))[0][0]
        #                     native_contacts[(tmp_bead_idx, tmp_bead_jdx)] = 1
        #             components_min_distances[cc_a, cc_b] = np.inf
        #             components_min_distances[cc_b, cc_a] = np.inf

        #         import pdb;pdb.set_trace()
        ## Remove isolated beads in native contacts
        return bonds, native_contacts

    def write_pdb(self, filename: str):
        from Bio.PDB import PDBIO

        io=PDBIO(use_model_flag=1)
        io.set_structure(self.spreadedCgStructure)
        io.save(filename)

    def view(self, min_radius = 2.0, max_radius = 8.0, bead_radius=2.0, out_script="/tmp/Adaptive_show_beads.cxc", view=True):
        """
        Visualiza los beads y la estructura original en ChimeraX.
        """

        try:
            import os  
            os.system("chimerax --version")
        except:
            error_message = (
            f"❌ Error: ChimeraX executable not found in system PATH.\n"
            "Please ensure ChimeraX is installed and its directory is added to your system's PATH."
            "If you do not have it, you can download it here: https://www.cgl.ucsf.edu/chimerax/download.html"
            )
            raise FileNotFoundError(error_message)
        
        beads = np.array([i for i in self.spreadedCgStructure.get_atoms()])
        chain_by_idx = np.array([i.get_parent().get_parent().get_id() for i in beads])
        model_by_idx = np.array([i.get_parent().get_parent().get_parent().get_id() for i in beads])
        
        ## Save the coarse grained molecule as a PDB file
        self.write_pdb("/tmp/adaptiveCG.pdb")

        ## Write the ChimeraX script to visualize the original structure and the beads
        with open(out_script, "w") as f:

            # Create a new molecule for beads (fake atoms)
            f.write("# Creating bead pseudo-atoms\n")
            f.write("close all\n")     # hide everything first
            f.write("# ChimeraX script to visualize beads and original structure\n")
            f.write(f"open {self.inputPDBfilePath}\n\n")
            f.write("hide atoms\n")
            f.write("show cartoon\n")
            f.write(f"open /tmp/adaptiveCG.pdb\n\n")
            f.write("style sphere\n")
            f.write("color bychain\n")

            bead_name_by_idx = np.array([i.get_name() for i in beads])
            f.write("style sphere\n")
            f.write("color bychain\n")
           
            drawn_bonds = []
            for i, j, _ in self.getForceField()["bonds"]["data"]:
                bond_str = "_".join(np.sort([i, j]).astype(str))
                if bond_str not in drawn_bonds:
                    f.write(f"distance #2/{chain_by_idx[i]}@{bead_name_by_idx[i]} #2/{chain_by_idx[j]}@{bead_name_by_idx[j]} color blue radius 0.2\n")
                    drawn_bonds.append(bond_str)
                # f.write(f"distance #2.{model_by_idx[i] + 1}/{chain_by_idx[i]}@{bead_name_by_idx[i]} #2.{model_by_idx[j] + 1}/{chain_by_idx[j]}@{bead_name_by_idx[j]} color blue radius 0.2\n")
                # import pdb;pdb.set_trace()
            drawn_native_contacts = []
            for i, j, _, _, _ in self.getForceField()["nativeContacts"]["data"]:
                nc_str = "_".join(np.sort([i, j]).astype(str))
                if nc_str in drawn_bonds:
                    print(nc_str, " already in bonds")
                elif nc_str not in drawn_native_contacts:
                    f.write(f"distance #2/{chain_by_idx[i]}@{bead_name_by_idx[i]} #2/{chain_by_idx[j]}@{bead_name_by_idx[j]} color red radius 0.2\n")
                    drawn_native_contacts.append(nc_str)

            # for tmp_data in self.getForceField()["bonds"]["data"]:
            #     i, j = tmp_data[0], tmp_data[1]
            #     f.write(f"distance #2/{chain_by_idx[i]}@{bead_name_by_idx[i]} #2/{chain_by_idx[j]}@{bead_name_by_idx[j]} color blue radius 0.2\n")
            #     # f.write(f"distance #2.{model_by_idx[i] + 1}/{chain_by_idx[i]}@{bead_name_by_idx[i]} #2.{model_by_idx[j] + 1}/{chain_by_idx[j]}@{bead_name_by_idx[j]} color blue radius 0.2\n")
            #     # import pdb;pdb.set_trace()
            # for tmp_data in self.getForceField()["nativeContacts"]["data"]:
            #     i, j = tmp_data[0], tmp_data[1]
            #     f.write(f"distance #2/{chain_by_idx[i]}@{bead_name_by_idx[i]} #2/{chain_by_idx[j]}@{bead_name_by_idx[j]} color red radius 0.2\n")
                # f.write(f"distance #2.{model_by_idx[i] + 1}/{chain_by_idx[i]}@{bead_name_by_idx[i]} #2.{model_by_idx[j] + 1}/{chain_by_idx[j]}@{bead_name_by_idx[j]} color red radius 0.2\n")

            f.write("show #1 cartoon\n\n")       # show original PDB
            f.write("hide #1 atoms\n\n")       # show original PDB

            f.write("lighting depthCue false\n")  # hide initial beads
            f.write("zoom\n")

        print(f"CXC script written to {out_script}")
        if view:
            import os
            os.system(f"chimerax {out_script} &")

