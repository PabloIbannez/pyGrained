import os
import json

import logging

def getData(dataName):

    logger = logging.getLogger("pyGrained")

    currentPath = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(currentPath,"..","data",f"{dataName}.json"),"r") as df:
            data = json.load(df)
        return data
    except:
        logger.critical(f"Data {dataName} not found")
        raise FileNotFoundError()

