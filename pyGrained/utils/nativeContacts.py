import os
import json

import logging

from scipy.spatial import cKDTree

from tqdm import tqdm

from .geometric import *

import pdb

#All structured are assumed to be spreaded
# index = atm.get_serial_number()

def generateNativeContactsByCutOff(atoms,ids,positions,cutOff,nexcl):

    logger = logging.getLogger("pyGrained")
    logger.info("Generating native contacts ...")

    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutOff)

    nativeContacts = {}
    for pair in tqdm(pairs):

        index1 = pair[0]
        index2 = pair[1]

        mdl1   = atoms[ids[index1]].get_parent().get_parent().get_parent().get_id()
        mdl2   = atoms[ids[index2]].get_parent().get_parent().get_parent().get_id()

        chain1 = atoms[ids[index1]].get_parent().get_parent().get_id()
        chain2 = atoms[ids[index2]].get_parent().get_parent().get_id()

        res1   = atoms[ids[index1]].get_parent().get_id()[1]
        res2   = atoms[ids[index2]].get_parent().get_id()[1]

        #print(mdl1,mdl2,chain1,chain2,res1,res2)

        if mdl1 == mdl2 and chain1 == chain2 and abs(res1 - res2) <= nexcl:
            continue
        else:
            nativeContacts[(ids[index1],ids[index2])] = {}

    return nativeContacts

def generateNativeContactsByCutOffFromStructure(structure,cutOff,nexcl):

    atoms = list(structure.get_atoms())

    ids       = [atm.get_serial_number() for atm in atoms]
    positions = [atm.get_coord()         for atm in atoms]

    return generateNativeContactsByCutOff(atoms,ids,positions,cutOff,nexcl)

def generateNativeContactsAlphaCarbonByCutOffFromStructure(structure,cutOff,nexcl):

    atoms = list(structure.get_atoms())

    ids       = [atm.get_serial_number() for atm in atoms if atm.get_name() == "CA"]
    positions = [atm.get_coord()         for atm in atoms if atm.get_name() == "CA"]

    return generateNativeContactsByCutOff(atoms,ids,positions,cutOff,nexcl)

def generateKaranicolasBrooksNativeContactsFromStructure(structure,cutOff,nexcl):
    #Expected structure to be all atom

    def areResiduesInContact(res1,res2,cutOff):

        def getAtoms(residue):
            return [atm for atm in residue.get_atoms() if atm.get_name() in ["N","CA","C","O"]]

        for atm1 in getAtoms(res1):
            for atm2 in getAtoms(res2):
                if atomDst(atm1,atm2) < cutOff:
                    return True

        return False

    def areResiduesHydrogenBonded(resi,resj):

        if(resj.get_resname() == "PRO"):
            return False
        #Check if resj is the first residue in the chain
        if(resj.get_id()[1] == 1):
            return False
        #Check if resi is the last residue in the chain
        if(resj.get_id()[1] == list(resj.get_parent().get_residues())[-1].get_id()[1]):
            return False

        q1 = 0.42;
        q2 = 0.20;
        f  = 332.0716;
        F  = q1*q2*f;

        for atm in resi.get_atoms():
            if atm.get_name() == "O":
                O = atm
            if atm.get_name() == "C":
                C = atm

        for atm in resj.get_atoms():
            if atm.get_name() == "N":
                N = atm
            if atm.get_name() == "H":
                H = atm

        r_ON = atomDst(O,N);
        r_CH = atomDst(C,H);
        r_OH = atomDst(O,H);
        r_CN = atomDst(C,N);

        e = F*(1.0/r_ON+1.0/r_CH-1.0/r_OH-1.0/r_CN);

        if(e < -0.5):
            return True
        else:
            return False


    #Alpha carbons close than cutOff, index refered to all atom
    caPairs = generateNativeContactsAlphaCarbonByCutOffFromStructure(structure,cutOff,nexcl)

    #Convert pairs to neighbours list
    indicesCA    = sorted(set(list([pair[0] for pair in caPairs] + [pair[1] for pair in caPairs])))
    neighboursCA = {index:[] for index in indicesCA}
    for pair in caPairs:
        neighboursCA[pair[0]].append(pair[1]) #Note no symmetry. We want to iterate over all pairs only once

    ############################################################

    currentPath = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(currentPath,"..","data","MiyazawaJernigaEnergyParm.json"),"r") as fe:
        interactionEnergy = json.load(fe)

    interactionEnergyTmp = {}
    for key in interactionEnergy:
        t1,t2 = key.split("_")
        interactionEnergyTmp[f"{t2}_{t1}"] = interactionEnergy[key]

    #Merge both dictionaries
    interactionEnergy.update(interactionEnergyTmp)

    ############################################################

    nativeContacts = {}

    atoms = list(structure.get_atoms())
    for index in tqdm(indicesCA):
        for neig in neighboursCA[index]:

            index1 = index
            index2 = neig

            res1   = atoms[index1].get_parent()
            res2   = atoms[index2].get_parent()

            res1Name = res1.get_resname()
            res2Name = res2.get_resname()

            res1Seq = res1.get_id()[1]
            res2Seq = res2.get_id()[1]

            chain1 = res1.get_parent().get_id()
            chain2 = res2.get_parent().get_id()

            mdl1   = res1.get_parent().get_parent().get_id()
            mdl2   = res1.get_parent().get_parent().get_id()

            differentChain = ((mdl1 != mdl2) or (chain1 != chain2))

            if not differentChain and abs(res1Seq - res2Seq) <= nexcl:
                continue
            else:
                c = 0
                h = 0

                nCType = ""
                nCEnergy = 0.0

                if areResiduesInContact(res1,res2,4.5):
                    nCType   += "S"
                    nCEnergy += interactionEnergy[f"{res1Name}_{res2Name}"]
                    c += 1

                if areResiduesHydrogenBonded(res1,res2):
                    nCType   += "H"
                    h += 1

                if areResiduesHydrogenBonded(res2,res1): #Note res2,res1
                    nCType   += "H"
                    h += 1

                if (c+h) > 0:
                    nativeContacts[(index1,index2)] = {"type":nCType,"energy":nCEnergy}

                if (c==0 and h>0):
                    nativeContacts[(index1,index2)]["energy"] += 1.0

                if (c+h) >= 2:

                    if(h==2):
                        bType = "O"
                        nadd  = 1
                    if(c+h==2):
                        bType = "O"
                        nadd  = 1
                    if(c+h==3):
                        bType = "OO"
                        nadd  = 2

                    res1_prev = None
                    for res in res1.get_parent().get_residues():
                        if res.get_id()[1] == res1Seq - 1:
                            res1_prev = res
                            break

                    res1_next = None
                    for res in res1.get_parent().get_residues():
                        if res.get_id()[1] == res1Seq + 1:
                            res1_next = res
                            break

                    res2_prev = None
                    for res in res2.get_parent().get_residues():
                        if res.get_id()[1] == res2Seq - 1:
                            res2_prev = res
                            break

                    res2_next = None
                    for res in res2.get_parent().get_residues():
                        if res.get_id()[1] == res2Seq + 1:
                            res2_next = res
                            break

                    orientationalHydrogenBondCount = 0

                    #res1_prev,res2
                    if res1_prev:
                        if abs(res1_prev.get_id()[1] - res2Seq) > nexcl or differentChain:
                            orientationalHydrogenBondCount += 1.0

                    #res1,res2_prev
                    if res2_prev:
                        if abs(res1Seq - res2_prev.get_id()[1]) > nexcl or differentChain:
                            orientationalHydrogenBondCount += 1.0

                    #res1_next,res2
                    if res1_next:
                        if abs(res1_next.get_id()[1] - res2Seq) > nexcl or differentChain:
                            orientationalHydrogenBondCount += 1.0

                    #res1,res2_next
                    if res2_next:
                        if abs(res1Seq - res2_next.get_id()[1]) > nexcl or differentChain:
                            orientationalHydrogenBondCount += 1.0

                    oe = float(nadd)/float(orientationalHydrogenBondCount)

                    #res1_prev,res2
                    if res1_prev:
                        if abs(res1_prev.get_id()[1] - res2Seq) > nexcl or differentChain:
                            neig1 = atoms.index(res1_prev["CA"])
                            neig2 = index2

                            if (neig1,neig2) not in nativeContacts:
                                nativeContacts[(neig1,neig2)] = {"type":bType,"energy":oe}
                            else:
                                nativeContacts[(neig1,neig2)]["type"]   += bType
                                nativeContacts[(neig1,neig2)]["energy"] += oe

                    #res1,res2_prev
                    if res2_prev:
                        if abs(res1Seq - res2_prev.get_id()[1]) > nexcl or differentChain:
                            neig1 = index1
                            neig2 = atoms.index(res2_prev["CA"])

                            if (neig1,neig2) not in nativeContacts:
                                nativeContacts[(neig1,neig2)] = {"type":bType,"energy":oe}
                            else:
                                nativeContacts[(neig1,neig2)]["type"]   += bType
                                nativeContacts[(neig1,neig2)]["energy"] += oe

                    #res1_next,res2
                    if res1_next:
                        if abs(res1_next.get_id()[1] - res2Seq) > nexcl or differentChain:
                            neig1 = atoms.index(res1_next["CA"])
                            neig2 = index2

                            if (neig1,neig2) not in nativeContacts:
                                nativeContacts[(neig1,neig2)] = {"type":bType,"energy":oe}
                            else:
                                nativeContacts[(neig1,neig2)]["type"]   += bType
                                nativeContacts[(neig1,neig2)]["energy"] += oe

                    #res1,res2_next
                    if res2_next:
                        if abs(res1Seq - res2_next.get_id()[1]) > nexcl or differentChain:
                            neig1 = index1
                            neig2 = atoms.index(res2_next["CA"])

                            if (neig1,neig2) not in nativeContacts:
                                nativeContacts[(neig1,neig2)] = {"type":bType,"energy":oe}
                            else:
                                nativeContacts[(neig1,neig2)]["type"]   += bType
                                nativeContacts[(neig1,neig2)]["energy"] += oe

    for (i,j) in nativeContacts:
        r0 = atomDst(atoms[i],atoms[j])
        nativeContacts[(i,j)]["r0"] = r0

    return nativeContacts
