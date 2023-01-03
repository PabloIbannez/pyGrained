# pyCoarse
pyCoarse is a Python library designed for generating coarse-grained models of proteins, particularly for the development of coarse-grained biomolecular models such as viruses.
# Coarse grained model generation
These steps are common to all coarse grained models.

 1. PDB file is load and analyzed. Different chains are grouped into classes. Two chains are considered to belong to the same class if one chain's sequence is contained within the other. Each class is associated with a leader chain, which is the shortest chain in the class.
 2. A structure is created containing only the leader chains. The rest of the protein is stored as a translation and rotation operation.
 3. The coarse-grained model is then applied to each leader chain, and the full structure is reconstructed at coarse-grained resolution.

 ## Coarse grained models availble
 ### Shape based coarse grained model (SBCG)
 More information about this model can be found [here](https://www.ks.uiuc.edu/Research/CG/sbcg.html).  The following example shows how to construct the coarse grained using the SBCG model. The P22 virus (PDB id : 5uu5) is used.
```
import pyGrained
import pyGrained.models

resolution = 300 #Number of atoms per bead
steps      = 10000 #Number of minimization steps

pathToPDB = "./p22/data/5uu5.pdb"
outputFile = "p22_sbcg"

test = pyGrained.models.SBCG("p22_sbcg",pathToPDB)
test.generateModel(resolution,steps)

#Models to be used in topology generation. Elastic network model for the bonds and a proximity model of alpha carbons for native contacts.

model = {"bondsModel":{"name":"ENM",
                       "parameters":{"enmCut":12.0}},
         "nativeContactsModel":{"name":"CA",
                                "parameters":{"ncCut":8.0}}
         }

test.generateTopology(model)

#Generate .json with all information
test.generateSimulation(outputFile+".json",K=5.0,epsilon=1.0,D=1.2)

#Write a PDB file with the coarse grained model
test.writePDBcg(outputFile+".pdb")

#Write a superpunto file
test.writeSPcg(outputFile+".sp")
```
