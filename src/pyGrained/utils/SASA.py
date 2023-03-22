import freesasa

def computeStructureSASA(structure):

    sasa,_   = freesasa.calcBioPDB(structure)
    sasaAtom = [sasa.atomArea(i) for i in range(sasa.nAtoms())]

    return sasaAtom

