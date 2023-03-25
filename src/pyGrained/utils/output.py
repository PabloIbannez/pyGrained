def writePDB(structure,outName):
    io=PDBIO(use_model_flag=1)
    io.set_structure(structure)
    io.save(outName)

def writePQR(structure,outName):
    io=PDBIO(use_model_flag=1,is_pqr=True)
    io.set_structure(structure)
    io.save(outName)

def writeSP(structure,outName):
    with open(outName,"w") as f:
        for bead in structure.get_atoms():
            pos = bead.get_coord()
            r   = bead.radius
            c   = int("".join([str(ord(i)) for i in bead.get_parent().get_parent().get_id()]))%256
            f.write(f"{pos[0]} {pos[1]} {pos[2]} {r} {c}\n")

