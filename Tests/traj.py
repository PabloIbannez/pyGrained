import os

from pyGrained.models.SBCG import SBCG

from pyGrained.utils.output import writeSP
from pyGrained.utils.trajectory import applyCoarseGrainedOverTrajectory

dataPath = "./data/"

resolution = 300
steps     = 10000

aliasFilepathMinBeads = [["p22",dataPath+"p22/5uu5.pdb"]]

# Check if folder results exists
if not os.path.exists("./results"):
    os.makedirs("./results")

for a,fl in aliasFilepathMinBeads:
    out = "./results/" + a

    model = {"SASA":False, #!!!
             "parameters":{"resolution":resolution,
                           "steps":steps,
                           "bondsModel":{"name":"ENM",
                                         "parameters":{"enmCut":12.0,
                                                       "K":10.0}},
                           "nativeContactsModel":{"name":"CA",
                                                  "parameters":{"ncCut":8.0,
                                                                "epsilon":-0.5,
                                                                "D":1.2}}}
             }
    test = SBCG(a,fl,model,debug=True)

    writeSP(test.getSpreadedCgStructure(),out+"_SBCG.sp")

    if a == "p22":

        trajPDB = dataPath + "p22/5uu5_hexon_traj/init_prot.pdb"
        trajDCD = dataPath + "p22/5uu5_hexon_traj/traj_prot.dcd"

        cgTraj = applyCoarseGrainedOverTrajectory(test.getSpreadedCgMap(),
                                                  trajPDB,trajDCD)

        N=len(cgTraj[0].keys())

        with open(out+"TrajTest.xyz","w") as out:
            for ts in cgTraj.keys():
                out.write(str(N)+"\n")
                out.write("*\n")
                for s in cgTraj[ts].keys():
                    name = str(s[0])+str(s[2])
                    x,y,z = cgTraj[ts][s]
                    out.write(f"{name} {x} {y} {z} \n")
