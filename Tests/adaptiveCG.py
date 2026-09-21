import os

import json
import jsbeautifier

from pyGrained.models.AdaptiveCG import AdaptiveCG

from pyGrained.utils.output import writeSP
from pyGrained.utils.output import types2global

pdb_file1 = "./data/lizard_6qi5_clean.pdb"  # Replace with your PDB file path
pdb_file2 = "./data/HAdV5_6b1t_clean.pdb"  # Replace with your PDB file path

params = {
    "parameters": {
        "resolution": 250, # Atoms per bead
        "bondsModel": {"name":"ENM", "parameters":{
            "enmCut": 20.0,      # Cutoff distance for the elastic network
            "K": 1.0,            # Common spring constant
        }},
        "nativeContactsModel":{"name":"cutOff", "parameters":{
            "ncCut": 20.0,       # Cutoff distance for the native contacts
            "epsilon": 1.0,      # Depth of the Morse well, per contact
            "D": 1.0,            # Width of the Morse well
            "eps0": 1.0,         # WCA epsilon, common to every contact
        }},
        },
    "SASA": False
}

params2 = {
    "parameters": {"resolution":200, 
                   "steps":1000, 
                   "bondsModel":{"name":"count"},
                   "nativeContactsModel":{"name":"CA", "parameters":{
                       "epsilon":1.0,
                       "D":1.0
                   }},
                   }, 
    "SASA": False
}

model = AdaptiveCG("test", pdb_file2, params=params)

writeSP(model.getSpreadedCgStructure(),"./data/hadv5_CG.sp")
# # model = SBCG("test", pdb_file1, params=params2)
# # model = AdaptiveCG("test", pdb_file, params=params)

# model = AdaptiveCG(pdb_file, n_beads, sigma)
# R_opt, chi_opt = model.optimize(max_iter=3000)

# pdb_file = "/home/pablo/Lizard_MD/structures/au_mcps/6qi5_mcp.pdb"  # Replace with your PDB file path
# unique_mols = model.compute_unique_molecules()

with open("HADV5_SOP.json", 'w') as outfile:
    glb        = types2global(model.getTypes())
    state      = model.getState()
    structure  = model.getStructure()
    forceField = model.getForceField()
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    top = {"structure":structure,"forceField":forceField}
    outfile.write(jsbeautifier.beautify(json.dumps({"global":glb,"state":state,"topology":top}), opts))
