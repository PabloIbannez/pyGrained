from pyGrained.models.SBCG import SBCG
from pyGrained.models.AlphaCarbon import KaranicolasBrooks

from pyGrained.utils.output import writeSP

resolution = 300 #Number of atoms per bead
steps      = 10000 #Number of minimization steps

pathToPDB = "./data/1egl/1egl.pdb"
outputFile = "1egl"

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
#
#test = SBCG("cox",pathToPDB,model,debug=True)

model = {}

test = KaranicolasBrooks("1egl",pathToPDB,model,debug=True)

writeSP(test.getSpreadedCgStructure(),"1egl.sp")


