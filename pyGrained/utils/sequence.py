from Bio.Data.PDBData import protein_letters_3to1

def getChainSequence(chain):
    seqDict = {res.get_id()[1]:protein_letters_3to1[res.get_resname()] for res in chain.get_residues()}

    seqId = sorted(list(seqDict.keys()))

    seq = ""
    for i in range(1,max(seqId)):
        if i in seqDict.keys():
            seq+=seqDict[i]
        else:
            seq+="X"

    return seq

