import logging

import MDAnalysis as mda

from tqdm import tqdm

def applyCoarseGrainedOverTrajectory(cgMap,trajPDB,trajDCD):

    logger = logging.getLogger("pyGrained")

    logger.info("Applying coarse grained map to trajectory")

    mdls = [bead[0] for bead in cgMap]
    m0   = min(mdls)
    m0CgMap = {bead:cgMap[bead] for bead in cgMap.keys() if bead[0] == m0}

    universe = mda.Universe(trajPDB,trajDCD)

    chains = [ch.segid for ch in universe.segments]

    cg2sel = {}
    for bead in tqdm(m0CgMap.keys()):
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

