import logging

import copy

def generateSpreadedCgMap(spreadedStructure,
                          classes,
                          aggregatedCgStructure,
                          spreadedCgStructure,
                          aggregatedCgMap):

    ch2leader = {}
    for clsName in classes:
        leader = classes[clsName]["leader"]
        ch2leader[leader]=[leader]
        for mmb in classes[clsName]["members"]:
            ch2leader[mmb]=leader

    atm2index = {}
    for atm in spreadedStructure.get_atoms():
        mdl_id = atm.get_parent().get_parent().get_parent().get_id()
        ch_id  = atm.get_parent().get_parent().get_id()
        res_id = atm.get_parent().get_id()[1]
        atm_id = atm.get_name()
        atm2index[(mdl_id,ch_id,res_id,atm_id)] = atm.get_serial_number()

    spreadedCgMap = {}

    mdl_cg = list(aggregatedCgStructure.get_models())[0].get_id()
    for bead in spreadedCgStructure.get_atoms():
        mdl_id = bead.get_parent().get_parent().get_parent().get_id()
        ch_id  = bead.get_parent().get_parent().get_id()
        res_id = bead.get_parent().get_id()[1]
        atm_id = bead.get_name()

        currentBead = (mdl_id,ch_id,res_id,atm_id,bead.get_serial_number())

        spreadedCgMap[currentBead] = []
        for atm in aggregatedCgMap[(mdl_cg,ch2leader[ch_id],res_id,atm_id)]:
            mdl_atm,ch_atm,res_atm,atm_atm = atm
            index = atm2index[(mdl_id,ch_id,res_atm,atm_atm)]
            spreadedCgMap[currentBead].append((mdl_id,ch_id,res_atm,atm_atm,index))

    return spreadedCgMap

def generateInverseIndexMap(map_):
    invMap = {}
    for key in map_.keys():
        for value in map_[key]:
            indexValue = value[-1]
            indexKey   = key[-1]

            invMap[indexValue] = indexKey

    return invMap

def generateTypes(spreadedCgStructure,SASA=False):

    #Types
    logger = logging.getLogger("pyGrained")

    logger.info(f"Generating types ...")

    #Generate types
    types={}
    for atm in list(spreadedCgStructure.get_models())[0].get_atoms():
        typeName = atm.get_name()

        if typeName not in types.keys():

            mass      = round(atm.mass,3)
            radius    = round(atm.radius,3)
            charge    = round(atm.get_charge(),3)

            if SASA:
                totalSASA = round(atm.totalSASA,3)
                totalSASApolar  = round(atm.totalSASApolar,3)
                totalSASAapolar = round(atm.totalSASAapolar,3)

            types[typeName]={"name":typeName,"mass":mass,"radius":radius,"charge":charge}

    logger.debug(f"Types: {types}")
    logger.info(f"Types generation end")

    #Types end
    return copy.deepcopy(types)

def generateState(spreadedCgStructure):

    #State
    logger = logging.getLogger("pyGrained")

    logger.info(f"Generating state ...")

    state = {}
    state["labels"] = ["id", "position"]
    state["data"]   = []

    for atm in spreadedCgStructure.get_atoms():
        state["data"].append([atm.get_serial_number(),list(atm.get_coord())])

    #logger.debug(f"State: {state}")
    logger.info(f"State generation end")

    #State end
    return state

def generateStructure(spreadedCgStructure):

    #Structure
    logger = logging.getLogger("pyGrained")

    logger.info(f"Generating structure ...")

    structure = {}
    structure["labels"] = ["id", "type", "modelId","chainId","resId"]
    structure["data"]   = []

    chains     = []
    for atm in spreadedCgStructure.get_atoms():
        #mdl = atm.get_parent().get_parent().get_parent().get_id()
        mdl = 0 #All atoms are in the same model
        ch  = atm.get_parent().get_parent().get_id()
        res = atm.get_parent().get_id()[1]

        if ch not in chains:
            chains.append(ch)

        chainIndex = chains.index(ch)

        structure["data"].append([atm.get_serial_number(),
                                  atm.get_name(),
                                  mdl,
                                  chainIndex,
                                  res])

    #logger.debug(f"Structure: {structure}")
    logger.info(f"Structure generation end")

    #Structure end
    return structure


