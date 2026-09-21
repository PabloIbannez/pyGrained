"""Adaptive coarse-grained (AdaptiveCG) model.

The bead mapping implemented in this module follows Sec. II A ("The CG
mapping") of

    C. Monago, J. A. de la Torre, R. Delgado-Buscalioni and P. Espanol,
    "Unraveling internal friction in a coarse-grained protein model",
    J. Chem. Phys. 162, 114115 (2025).
    https://doi.org/10.1063/5.0255498

All equation numbers quoted in this module refer to that paper.
"""

from .. import CoarseGrainedBase

import os
import warnings
import itertools

import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

from Bio.PDB import Structure, Model, Chain, Residue, Atom

from ..utils.atomList import *
from ..utils.coarseGrained import *

class AdaptiveCG(CoarseGrainedBase):

    def __beadPairs(self,cgStructure,cutOff,condition):
        """Bead pairs closer than cutOff, as {(id_i,id_j):r0}.

        The ids are the bead serial numbers of the spreaded CG structure,
        which run from 0 in the order the beads are iterated.

        condition keeps the pairs within the same chain ("intra"), the pairs
        in different chains ("inter") or every pair ("all"). Two beads are in
        the same chain when both their model and their chain ids match.
        """

        if condition not in ("intra","inter","all"):
            self.logger.error(f"Unknown condition {condition}, expected intra, inter or all")
            raise Exception("Unknown condition for the bead pairs")

        beads  = list(cgStructure.get_atoms())
        ids    = [bead.get_serial_number() for bead in beads]
        mdlCh  = [(bead.get_parent().get_parent().get_parent().get_id(),
                   bead.get_parent().get_parent().get_id()) for bead in beads]
        coords = np.asarray([bead.get_coord() for bead in beads])

        kd = cKDTree(coords)

        pairs = {}
        for i,j in kd.query_pairs(cutOff):

            sameChain = mdlCh[i] == mdlCh[j]

            if condition == "intra" and not sameChain:
                continue
            if condition == "inter" and sameChain:
                continue

            pairs[(ids[i],ids[j])] = float(np.linalg.norm(coords[i]-coords[j]))

        return pairs

    def __generateENM(self,cgStructure,enmCut,condition):
        """Bonds: an elastic network between every pair of beads within enmCut."""

        bonds = self.__beadPairs(cgStructure,enmCut,condition)

        self.logger.info(f"Generated {len(bonds)} ENM bonds ({condition}) with a cut off of {enmCut}")

        return bonds

    def __generateNativeContacts(self,cgStructure,ncCut,condition):
        """Native contacts: every pair of beads within ncCut."""

        nativeContacts = self.__beadPairs(cgStructure,ncCut,condition)

        self.logger.info(f"Generated {len(nativeContacts)} native contacts ({condition}) with a cut off of {ncCut}")

        return nativeContacts

    def _mapping(self, positions, masses, resolution, minBeads, seed):
        """Bead positions for one chain: mass-weighted k-means.

        The paper places the beads by minimizing the Kullback-Leibler
        divergence between the atomistic and the coarse-grained mass
        densities (Eqs. 2-7), which yields the fixed-point condition

            R_mu = sum_i m_i r_i chi_imu / sum_i m_i chi_imu          (Eq. 8)

        with chi the Shepard functions (Eq. 9)

            chi_imu = Delta(r_i - R_mu) / sum_nu Delta(r_i - R_nu)

        and Delta a Gaussian of width sigma (Eq. 3), solved by the iteration
        of Eq. 11.

        This is equivalent to a mass-weighted k-means in the limit sigma -> 0.
        There the Shepard functions become the characteristic functions of the
        Voronoi cells around R_mu, so the assignment step reduces to picking
        the nearest bead and Eq. 11 becomes Lloyd's algorithm on the objective

            J = sum_i m_i |r_i - R_mu(i)|^2 = sum_mu M_mu Rg_mu^2

        i.e. it makes the beads as compact as possible in a mass-weighted
        sense. That limit is the regime the paper works in: it picks
        sigma = 0.053, about 0.01 times the bead spacing, precisely so that
        the mapping delta_mu_i of Eq. 1 is 0 or 1 and the atoms of a bead are
        always the same. So k-means is used here directly.

        At finite sigma the same procedure gives fractional assignments (soft
        k-means, the EM algorithm for an isotropic Gaussian mixture of fixed
        width sigma). That is a generalization of the paper and is not
        implemented for now.
        """

        Nall   = positions.shape[0]
        Nbeads = int(Nall/resolution)+1

        self.logger.info(f"Generating AdaptiveCG mapping, from {Nall} atoms to {Nbeads} beads")

        if Nbeads <= minBeads:
            return []

        # tol=0 makes KMeans stop only once the assignment stops changing, so
        # that each center really is the mass-weighted centroid of its own
        # Voronoi cell, which is Eq. 8 in this limit. The default tolerance is
        # relative to the variance of the data and leaves the centers off that
        # fixed point (by ~0.1 A on a protein-sized chain).
        #
        # random_state is fixed so that the mapping is reproducible: Lloyd's
        # algorithm only reaches a local minimum of J, so different seeds give
        # slightly different bead positions.
        kmeans = KMeans(n_clusters=Nbeads, n_init=10, tol=0.0, random_state=seed)
        kmeans.fit(positions, sample_weight=masses)

        return kmeans.cluster_centers_.astype(float)

    def __init__(self,
                 name:str,
                 inputPDBfilePath:str,
                 params:dict,
                 debug = False):

        self.inputPDBfilePath = os.path.abspath(inputPDBfilePath)

        SASA = params.get("SASA",False)

        super().__init__(tpy  = "AdaptiveCG",
                         name = name,
                         inputPDBfilePath = inputPDBfilePath,
                         removeHetatm = True, removeHydrogens = False, removeNucleics = True,
                         centerInput = params.get("centerInput",True),
                         SASA = SASA,
                         aggregateChains = params.get("aggregateChains",True),
                         debug = debug)

        #We have to set types,states,structure and forceField

        #####################################################
        ################### GENERATE MODEL ##################

        self.logger.info(f"Generating coarse grained model (AdaptiveCG) ...")

        globalParams = params["parameters"]

        resolution = globalParams["resolution"]
        minBeads   = globalParams.get("minBeads",1)
        seed       = globalParams.get("seed",0)

        self.SASA = SASA

        aggregatedCgMap = {}
        spreadedCgMap   = {}

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

                    else:
                        continue

                    self.logger.info(f"Working in class {clsName} which leader is {chName}.")
                    positions_cg = self._mapping(positions,masses,resolution,minBeads,seed)
                    Ncg = len(positions_cg)

                    ##########################
                    #Voronoi

                    if Ncg > 0:

                        ch_cg = Chain.Chain(ch.get_id())
                        mdl_cg.add(ch_cg)

                        # Hard nearest-bead assignment: delta_mu_i of Eq. 1,
                        # i.e. the paper's mapping in the sigma -> 0 limit.
                        kd = cKDTree(positions_cg)
                        allIndex2cgIndex = kd.query(positions)[1]

                        cgIndex2allAtoms = [[] for _ in range(Ncg)]
                        for allIndex,cgIndex in enumerate(allIndex2cgIndex):
                            cgIndex2allAtoms[cgIndex].append(chAtoms[allIndex])

                        for cgIndex in range(Ncg):

                            atmList = cgIndex2allAtoms[cgIndex]

                            if not atmList:
                                self.logger.warning(f"Bead {cgIndex} of class {clsName} got no atoms assigned. Ignoring it.")
                                continue

                            ##########################

                            chName = self.getClasses()[clsName]["leader"]

                            cgName   = chName+str(cgIndex)
                            # At a Lloyd fixed point the mass-weighted centroid
                            # of the Voronoi cell is the k-means center, so this
                            # is positions_cg[cgIndex] up to the solver tolerance.
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
            enmCut    = bondsModel["parameters"]["enmCut"]
            condition = bondsModel["parameters"].get("condition","intra")
            bonds     = self.__generateENM(spreadedCgStructure,enmCut,condition)
        else:
            self.logger.error(f"Bonds model {bondsModelName} is not availble")
            raise Exception(f"Bonds model not available")

        self.logger.info(f"Generating native contacts ...")

        nativeContacsModelName = nativeContactsModel["name"]
        if nativeContacsModelName == "cutOff":
            ncCut          = nativeContactsModel["parameters"]["ncCut"]
            condition      = nativeContactsModel["parameters"].get("condition","inter")
            nativeContacts = self.__generateNativeContacts(spreadedCgStructure,ncCut,condition)
        else:
            self.logger.error(f"Native contacts model {nativeContacsModelName} is not availble")
            raise Exception(f"Native contacts model not available")

        self.logger.info(f"Topology generation end")

        #############################################################

        #ForceField

        self.logger.info(f"Generating force field ...")

        forceField = {}

        #Bonds
        if bondsModelName == "ENM":
            forceField["bonds"] = {}
            forceField["bonds"]["type"]       = ["Bond2","HarmonicCommon_K"]
            forceField["bonds"]["parameters"] = {"K":bondsModel["parameters"]["K"]}
            forceField["bonds"]["labels"]     = ["id_i", "id_j", "r0"]
            forceField["bonds"]["data"]       = []

            for bnd,r0 in bonds.items():
                id_i,id_j = bnd
                forceField["bonds"]["data"].append([id_i,id_j,round(r0,3)])
        else:
            self.logger.error(f"Bonds model {bondsModelName} is not availble")
            raise Exception(f"Bonds model not available")

        #Native contacts
        if nativeContacsModelName == "cutOff":
            forceField["nativeContacts"] = {}
            forceField["nativeContacts"]["type"]       = ["Bond2","MorseWCACommon_eps0"]
            forceField["nativeContacts"]["parameters"] = {"eps0":nativeContactsModel["parameters"].get("eps0",1.0)}
            forceField["nativeContacts"]["labels"]     = ["id_i", "id_j", "r0"]
            forceField["nativeContacts"]["data"]       = []

            for nc,r0 in nativeContacts.items():
                id_i,id_j = nc
                forceField["nativeContacts"]["data"].append([id_i,id_j,round(r0,3)])
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

        self.logger.info(f"Force field generation end")

        #############################################################

        self.setAggregatedCgStructure(aggregatedCgStructure)
        self.setSpreadedCgStructure(spreadedCgStructure)
        self.setAggregatedCgMap(aggregatedCgMap)
        self.setSpreadedCgMap(spreadedCgMap)

        self.setTypes(types)
        self.setState(state)
        self.setStructure(structure)
        self.setForceField(forceField)
