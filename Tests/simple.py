import os

import json
import jsbeautifier

from pyGrained.models.SBCG import SBCG
from pyGrained.models.AlphaCarbon import SelfOrganizedPolymer
from pyGrained.models.AlphaCarbon import KaranicolasBrooks

from pyGrained.utils.output import writeSP
from pyGrained.utils.output import types2global

pathToPDB = "./data/3dkt/3dkt.pdb"
outputFile = "./results/3dkt"

# Check if folder results exists
if not os.path.exists("./results"):
    os.makedirs("./results")

# SBCG
print("Starting SBCG")
resolution = 300 #Number of atoms per bead
steps      = 10000 #Number of minimization steps

model = {"SASA":False,
         "parameters":{"resolution":resolution,
                       "steps":steps,
                       "bondsModel":{"name":"count"},
                       "nativeContactsModel":{"name":"count"}
         }}

testSBCG = SBCG("p22",pathToPDB,model,debug=True)
writeSP(testSBCG.getSpreadedCgStructure(),outputFile+"_SBCG.sp")

with open(outputFile+"_SBCG.json", 'w') as outfile:
    structure  = testSBCG.getStructure()
    forceField = testSBCG.getForceField()
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    outfile.write(jsbeautifier.beautify(json.dumps({"structure":structure,"forceField":forceField}), opts))

## Self-Organized Polymer
#
#model = {"parameters":{"epsilonNC":2.0}}
#
#print("Starting Self-Organized Polymer")
#testSOP = SelfOrganizedPolymer("tmv",pathToPDB,model,debug=True)
#writeSP(testSOP.getSpreadedCgStructure(),outputFile+"_SOP.sp")
#
#with open(outputFile+"_SOP.json", 'w') as outfile:
#    glb        = types2global(testSOP.getTypes())
#    state      = testSOP.getState()
#    structure  = testSOP.getStructure()
#    forceField = testSOP.getForceField()
#    opts = jsbeautifier.default_options()
#    opts.indent_size = 2
#    top = {"structure":structure,"forceField":forceField}
#    outfile.write(jsbeautifier.beautify(json.dumps({"global":glb,"state":state,"topology":top}), opts))
#
### Karanicolas-Brooks
##print("Starting Karanicolas-Brooks")
##model = {}
##testKB = KaranicolasBrooks("cox",pathToPDB,model,debug=True)
##writeSP(testKB.getSpreadedCgStructure(),outputFile+"_KB.sp")
#
#
#
