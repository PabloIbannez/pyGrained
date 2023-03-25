import numpy as np

def atomDst(atm1, atm2):

    coord1 = atm1.get_coord()
    coord2 = atm2.get_coord()

    return np.linalg.norm(coord1 - coord2)

def atomAngle(atm1, atm2, atm3):

    coord1 = atm1.get_coord()
    coord2 = atm2.get_coord()
    coord3 = atm3.get_coord()

    dr21 = coord1 - coord2
    dr23 = coord3 - coord2

    dr21 /= np.linalg.norm(dr21)
    dr23 /= np.linalg.norm(dr23)

    return np.arccos(np.dot(dr21, dr23))

def atomDihedral(atm1, atm2, atm3, atm4):

    coord1 = atm1.get_coord()
    coord2 = atm2.get_coord()
    coord3 = atm3.get_coord()
    coord4 = atm4.get_coord()

    dr12 = coord2 - coord1
    dr23 = coord3 - coord2
    dr34 = coord4 - coord3

    va = np.cross(dr12, dr23)
    vb = np.cross(dr23, dr34)

    va /= np.linalg.norm(va)
    vb /= np.linalg.norm(vb)

    sign = np.sign(np.dot(np.cross(dr12, dr23), dr34))

    return sign*np.arccos(np.dot(va, vb))

