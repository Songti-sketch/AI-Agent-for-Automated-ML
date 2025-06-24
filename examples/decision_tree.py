import os
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"

import pandas as pd
import numpy as np

from ml_master.agents import decision_tree_implementor

df = pd.read_csv("./resources/data.csv")

ret = decision_tree_implementor.invoke(
    df, 
    data_file_name="./resources/data.csv",
    target_column="Price", # predicting the price of the car
    metric_name="R2", # using R2 to eval
)

print("Code from AI:")
for line in ret.split("\n"):
    print(line)
print("\n\n\n")

# %%

from ml_master.misc import exec_code

ret_code, ret_str = exec_code(ret)

print("ret_code:", ret_code)
print(ret_str)

# %%
from ml_master.agents import refiner

print("--- refine ---")

refined_code = refiner.invoke(
    df,
    target_column="Price", # predicting the price of the car
    metric_name="R2", # using R2 to eval
    code_str=ret,
    exec_result=ret_str,
)

if refined_code != "":
    print(refined_code)
    ret_code, ret_str2 = exec_code(refined_code)

    print("ret_code:", ret_code)
    print(ret_str2)
    
# %%
from ml_master.agents import judge

print("--- judge ---")

ret_strs = [
    ret_str,
    ret_str2
]

better_idx = judge.invoke(
    ret_strs
)

print(f"index={better_idx} is better.")