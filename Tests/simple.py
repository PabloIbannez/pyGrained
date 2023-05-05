from pyGrained.models.SBCG import SBCG
from pyGrained.models.AlphaCarbon import SelfOrganizedPolymer
from pyGrained.models.AlphaCarbon import KaranicolasBrooks

from pyGrained.utils.output import writeSP

pathToPDB = "./data/cox/5xs5_fixed.pdb"
outputFile = "cox"

# SBCG
resolution = 300 #Number of atoms per bead
steps      = 10000 #Number of minimization steps

#model = {"SASA":True,
#         "parameters":{"resolution":resolution,
#                       "steps":steps,
#                       "bondsModel":{"name":"ENM",
#                                     "parameters":{"enmCut":12.0,
#                                                   "K":10.0}},
#                       "nativeContactsModel":{"name":"CA",
#                                              "parameters":{"ncCut":8.0,
#                                                            "epsilon":-0.5,
#                                                            "D":1.2}}}
#         }
#testSBCG = SBCG("cox",pathToPDB,model,debug=True)
#writeSP(testSBCG.getSpreadedCgStructure(),outputFile+"_SBCG.sp")
#
## Self-Organized Polymer
#model = {}
#testSOP = SelfOrganizedPolymer("cox",pathToPDB,model,debug=True)
#writeSP(testSOP.getSpreadedCgStructure(),outputFile+"_SOP.sp")

# Karanicolas-Brooks
model = {}
testKB = KaranicolasBrooks("cox",pathToPDB,model,debug=True)
writeSP(testKB.getSpreadedCgStructure(),outputFile+"_KB.sp")



