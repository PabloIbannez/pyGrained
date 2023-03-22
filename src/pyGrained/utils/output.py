def writePDBcg(self,outName):
    self.log.info(f"Writing pdb")
    io=PDBIO(use_model_flag=1)
    io.set_structure(self.spreadedCG)
    io.save(outName)

def writePQRcg(self,outName):
    self.log.info(f"Writing pqr")
    io=PDBIO(use_model_flag=1,is_pqr=True)
    io.set_structure(self.spreadedCG)
    io.save(outName)

def writeSPcg(self,outName):
    self.log.info(f"Writing sp")
    with open(outName,"w") as f:
        for bead in self.spreadedCG.get_atoms():
            pos = bead.get_coord()
            r   = bead.radius
            c   = hash(bead.get_parent().get_parent().get_id())%256
            f.write(f"{pos[0]} {pos[1]} {pos[2]} {r} {c}\n")

