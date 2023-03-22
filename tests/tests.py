import os

import pyGrained
import pyGrained.models

dataPath = "./data/"

resolution = 300
steps     = 10000

#aliasFilepathMinBeads = [["ubiq", dataPath+"ubiquitin/36b689z4lb.pqr",0],
#                         ["cox",  dataPath+"cox/5xs5.pdb",5],
#                         ["p22",  dataPath+"p22/5uu5.pdb",5],
#                         ["adeno",dataPath+"adeno/6cgv.pdb",10]]

aliasFilepathMinBeads = [["p22",dataPath+"p22/5uu5.pdb",5]]

outputFile = "testResults"
outputFileExist = os.path.exists(outputFile)
if not outputFileExist:
    os.makedirs(outputFile)

for a,fl,mB in aliasFilepathMinBeads:
    out = outputFile + "/" + a
    test = pyGrained.models.SBCG(out+"Test",fl,debug=False)
    test.generateModel(resolution,steps,minBeads=mB)

    model = {"bondsModel":{"name":"ENM",
                           "parameters":{"enmCut":12.0}},
             "nativeContactsModel":{"name":"CA",
                                    "parameters":{"ncCut":8.0}}
            }

    test.generateTopology(model)

    test.generateSimulation(out+"SimulationSbcgTest.json",5.0,1.0,1.2)
    test.writePDBcg(out+"SbcgTest.pdb")
    test.writeSPcg(out+"SbcgTest.sp")

    if a == "p22":

        trajPDB = dataPath + "p22/5uu5_hexon_traj/init_prot.pdb"
        trajDCD = dataPath + "p22/5uu5_hexon_traj/traj_prot.dcd"

        cgTraj = test.applyCoarseGrainedOverTrajectory(trajPDB,trajDCD)

        N=len(cgTraj[0].keys())

        with open(out+"TrajTest.xyz","w") as out:
            for ts in cgTraj.keys():
                out.write(str(N)+"\n")
                out.write("*\n")
                for s in cgTraj[ts].keys():
                    name = str(s[0])+str(s[2])
                    x,y,z = cgTraj[ts][s]
                    out.write(f"{name} {x} {y} {z} \n")
