# Amirreza Azari
# 99101087

import json
import math


def verifying_dm(inputs):
    n = len(inputs)
    inputs.sort(key=lambda inp: inp["deadline"])

    for inp in range(n):
        I_i = sum(inputs[j]["executionTime"] for j in range(inp))
        C_i = inputs[inp]["executionTime"]
        D_i = inputs[inp]["deadline"]
        R_i = 0
        while I_i + C_i > R_i:
            R_i = I_i + C_i
            if R_i > D_i:
                return False
            I_i = sum(math.ceil(R_i / inputs[j]["period"]) * inputs[j]["executionTime"] for j in range(inp))
    return True


def verifying_rm(inputs):
    n = len(inputs)
    u = 0
    for inp in inputs:
        u += inp["executionTime"] / inp["period"]
    upper_bound = n * (2 ** (1 / n) - 1)
    if upper_bound < u < 1:
        return None
    elif u <= upper_bound:
        return True
    else:
        return False


with open('input.json', 'r') as file:
    inputs = json.load(file)

result = {}
result["rm"] = verifying_rm(inputs)
result["dm"] = verifying_dm(inputs)

file = open("output.json", "w")
try:
    file.write(json.dumps(result))
finally:
    file.close()
