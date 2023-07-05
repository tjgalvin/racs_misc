#!/usr/bin/env python3

import sys
import traceback

import pandas as pd

def sex2deg(str):
    deg = 0.0
    negative = False
    for elem in reversed(str.split(":")):
        if negative:
            raise Exception("FFS")
            exit(1)
        deg /= 60.0
        f = float(elem)
        if f < 0.0:
            negative = True
            deg += abs(f)
        else:
            deg += f
    if negative:
        return -deg
    else:
        return deg

def sex2deg2(str):
    split = list(map(float, str.split(".")))
    if split[0] < 0.0:
        negative = True
    else:
        negative = False
    deg = abs(split[0])
    deg += split[1] / 60
    deg += split[2] / 3600
    num_decimal_digits = len(str.split(".")[3])
    split[3] /= 10**num_decimal_digits
    deg += split[3] / 3600
    if negative:
        return -deg
    else:
        return deg

df = pd.read_csv(sys.argv[1])
source_list = {}
for i, row in df.iterrows():
    if row["Type"] == "GAUSS":
        comp_type = {
            "gaussian": {
                "maj": row["MajorAxis"],
                "min": row["MinorAxis"],
                "pa": row["Orientation"],
            }
        }
    elif row["Type"] == "POINT":
        comp_type = "point"
    else:
        raise Exception(f'Unrecognised component type \'{row["Type"]}\'')

    flux_type = {
        "power_law": {
            #"si": row["SpectralIndex"],
            "si": -0.83,
            "fd": {
                #"freq": row["ReferenceFrequency"],
                "freq": 888500000.0,
                "i": row["I"],
            }
        }
    }
        
    source_list[row["Name"]] = [{
        "ra": sex2deg(row["Ra"]) * 15.0, # hours in 2023 lmao
        "dec": sex2deg2(row["Dec"]),
        "comp_type": comp_type,
        "flux_type": flux_type,
    }]
    #print(source_list)
    #break

import yaml
with open("hyp.yaml", "w") as h:
    yaml.dump(source_list, stream=h)