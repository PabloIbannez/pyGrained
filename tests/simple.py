from pyGrained.models.SBCG import SBCG

resolution = 300 #Number of atoms per bead
steps      = 10000 #Number of minimization steps

pathToPDB = "./data/p22/5uu5.pdb"
outputFile = "cox"


model = {"parameters":{"resolution":resolution,
                       "steps":steps,
                       "bondsModel":{"name":"ENM",
                                     "parameters":{"enmCut":12.0}},
                       "nativeContactsModel":{"name":"CA",
                                              "parameters":{"ncCut":8.0}}}
         }

test = SBCG("cox",pathToPDB,model,debug=True)
#
#
#
#
#test.generateModel(resolution,steps)
#
##Models to be used in topology generation. Elastic network model for the bonds and a proximity model of alpha carbons for native contacts.
#
#test.generateTopology(model)
#
##Generate .json with all information
#test.generateSimulation(outputFile+".json",K=5.0,epsilon=1.0,D=1.2)
#
##Write a PDB file with the coarse grained model
#test.writePDBcg(outputFile+".pdb")
#
##Write a superpunto file
#test.writeSPcg(outputFile+".sp")
