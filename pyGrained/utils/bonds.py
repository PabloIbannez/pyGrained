import logging

from scipy.spatial import cKDTree

from tqdm import tqdm

def generateBondsFromStructure(structure):

    logger = logging.getLogger("pyGrained")
    logger.info("Generating bonds ...")

    bonds = []

    for ch in structure.get_chains():
        atoms = list(ch.get_atoms())
        for atm1,atm2 in zip(atoms,atoms[1:]):
            id1 = atm1.get_serial_number()
            id2 = atm2.get_serial_number()
            if id2 - id1 != 1:
                logger.warning("Non-consecutive atom numbers in chain %s: %d and %d" % (ch.get_id(),id1,id2))
            else:
                bonds.append((id1,id2))

    #Check that for each atom id1 < id2.
    #If not, swap the ids and trhow a warning.
    for i,(id1,id2) in enumerate(bonds):
        if id1 > id2:
            bonds[i] = (id2,id1)
            logger.warning("Swapped atom ids in bond %d-%d" % (id1,id2))

    #Sort bonds by atom id
    bonds.sort(key=lambda x: x[0])

    #Check for duplicates
    for (id1,id2),(id3,id4) in zip(bonds,bonds[1:]):
        if id1 == id3 and id2 == id4:
            logger.warning("Duplicate bond found: %d-%d" % (id1,id2))

    #Remove duplicates
    bonds = list(set(bonds))

    return bonds

def generateAnglesFromBonds(bonds):

    logger = logging.getLogger("pyGrained")
    logger.info("Generating angles ...")

    ids = [id1 for id1,id2 in bonds] + [id2 for id1,id2 in bonds]
    ids = list(set(ids))
    ids.sort()

    id2bonds = {}
    for id1,id2 in bonds:
        id2bonds.setdefault(id1,[]).append(id2)
        id2bonds.setdefault(id2,[]).append(id1)

    angles = []
    for id1 in ids:
        for id2 in id2bonds[id1]:
            for id3 in id2bonds[id2]:
                if id3 != id1:
                    if id1 < id3:
                        angles.append((id1,id2,id3))
                    else:
                        angles.append((id3,id2,id1))

    angles = list(set(angles))

    #Sort angles by first atom id
    angles.sort(key=lambda x: x[0])

    return angles

def generateDihedralsFromBonds(bonds):

    logger = logging.getLogger("pyGrained")
    logger.info("Generating dihedrals ...")

    ids = [id1 for id1,id2 in bonds] + [id2 for id1,id2 in bonds]
    ids = list(set(ids))
    ids.sort()

    id2bonds = {}
    for id1,id2 in bonds:
        id2bonds.setdefault(id1,[]).append(id2)
        id2bonds.setdefault(id2,[]).append(id1)

    dihedrals = []
    for id1 in ids:
        for id2 in id2bonds[id1]:
            for id3 in id2bonds[id2]:
                if id3 != id1:
                    for id4 in id2bonds[id3]:
                        if id4 != id2 and id4 != id1: #Rings are not allowed
                            if id1 < id4:
                                dihedrals.append((id1,id2,id3,id4))
                            else:
                                dihedrals.append((id4,id3,id2,id1))

    dihedrals = list(set(dihedrals))

    #Sort dihedrals by first atom id
    dihedrals.sort(key=lambda x: x[0])

    return dihedrals

def generateElasticNetworkModel(ids,positions,cutOff):
    # If two atoms are closer than cutOff, they are bonded.
    # Atom with ids[i] is at position positions[i]

    logger = logging.getLogger("pyGrained")

    bonds = []

    logger.info("Generating elastic network model with cut-off %f" % cutOff)

    tree = cKDTree(positions)
    for i in tqdm(range(len(ids))):
        id1 = ids[i]
        pos1 = positions[i]
        for id2,dist in zip(*tree.query(pos1,cutOff)):
            if id1 < id2:
                bonds.append((id1,id2))

    bonds = list(set(bonds))

    #Sort bonds by atom id
    bonds.sort(key=lambda x: x[0])

    return bonds

def generateElasticNetworkModelFromStructure(structure,cutOff):
    ids = [atm.get_serial_number() for atm in structure.get_atoms()]
    positions = [atm.get_coord() for atm in structure.get_atoms()]
    return generateElasticNetworkModel(ids,positions,cutOff)

def generateAlphaCarbonElaticNetworkModelFromStructure(structure,cutOff):
    ids = [atm.get_serial_number() for atm in structure.get_atoms() if atm.get_name() == "CA"]
    positions = [atm.get_coord() for atm in structure.get_atoms() if atm.get_name() == "CA"]
    return generateElasticNetworkModel(ids,positions,cutOff)

def generateAlphaCarbonElaticNetworkModelFromStructureForEveryChain(structure,cutOff):
    bonds = []
    for ch in structure.get_chains():
        ids = [atm.get_serial_number() for atm in ch.get_atoms() if atm.get_name() == "CA"]
        positions = [atm.get_coord() for atm in ch.get_atoms() if atm.get_name() == "CA"]
        bonds += generateElasticNetworkModel(ids,positions,cutOff)
    return bonds

