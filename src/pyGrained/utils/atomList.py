import os

import json

import logging

import numpy as np

import freesasa

def computeAtomListCOM(atmList):
    masses = []
    wPos   = []
    for atm in atmList:
        masses.append(atm.mass)
        wPos.append(atm.get_coord()*atm.mass)
    masses = np.asarray(masses)
    wPos   = np.asarray(wPos)

    totalMass = np.sum(masses)

    return np.sum(wPos,axis=0)/totalMass

def computeAtomListMass(atmList):
    masses = []
    for atm in atmList:
        masses.append(atm.mass)
    masses = np.asarray(masses)

    return np.sum(masses)

def computeAtomListMassFromResidues(atmList):

    try:

        computeAtomListMassFromResidues.aminoacidMasses

    except:

        currentPath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(currentPath,"..","data","aminoacidMasses.json"),"r") as fchg:
            computeAtomListMassFromResidues.aminoacidMasses = json.load(fchg)

    mass = 0.0
    for atm in atmList:
        if atm.get_name() == "CA":
            mass+=computeAtomListMassFromResidues.aminoacidMasses[atm.get_parent().get_resname()]

    return mass

def computeAtomListCharge(atmList):
    charges = []
    for atm in atmList:
        chg = atm.get_charge()
        if chg == None:
            chg=0.0
        charges.append(chg)
    charges = np.asarray(charges)

    return np.sum(charges)

def computeAtomListChargeFromResidues(atmList):

    try:

        computeAtomListChargeFromResidues.aminoacidCharges

    except:

        currentPath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(currentPath,"..","data","aminoacidCharges.json"),"r") as fchg:
            computeAtomListChargeFromResidues.aminoacidCharges = json.load(fchg)

    charge = 0.0
    for atm in atmList:
        if atm.get_name() == "CA":
            charge+=computeAtomListChargeFromResidues.aminoacidCharges[atm.get_parent().get_resname()]

    return charge

def computeAtomListRadiusOfGyration(atmList):

    COM = computeAtomListCOM(atmList)
    M   = computeAtomListMass(atmList)

    wR2 = []
    for atm in atmList:
        R=np.linalg.norm(atm.get_coord()-COM)
        wR2.append(R*R*atm.mass)
    wR2 = np.asarray(wR2).sum()

    return np.sqrt(wR2/M)

def computeAtomListRadiusFromResidues(atmList):

    logger = logging.getLogger("pyGrained")

    try:

        computeAtomListRadiusFromResidues.aminoacidRadii

    except:

        currentPath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(currentPath,"..","data","aminoacidRadii.json"),"r") as fchg:
            computeAtomListRadiusFromResidues.aminoacidRadii = json.load(fchg)

    if len(atmList) > 1:
        logger.error("Computing the radius for a list of atoms longer than 1 is not implemented")
        raise NotImplementedError("Computing the radius for a list of atoms longer than 1 is not implemented")
    else:
        atm = atmList[0]
        return computeAtomListRadiusFromResidues.aminoacidRadii[atm.get_parent().get_resname()]

def computeAtomListSASA(atmList):

    logger = logging.getLogger("pyGrained")

    classifier = freesasa.Classifier()

    sasaPolar  = 0.0
    sasaApolar = 0.0

    for atm in atmList:
        tpy = classifier.classify(atm.get_parent().get_resname(),atm.get_name())
        if   tpy == "Polar":
            sasaPolar+=atm.totalSASA
        elif tpy == "Apolar":
            sasaApolar+=atm.totalSASA
        else:
            logger.critical(f"The type of the atom ({atm.get_parent().get_resname()},{atm.get_name()} is {tpy}. But it has to be \"Polar\" or \"Apolar\")")
            sys.exit(1)

    return sasaPolar,sasaApolar

