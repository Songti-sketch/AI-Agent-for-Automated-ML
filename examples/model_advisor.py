import os
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"

import pandas as pd
import numpy as np

from ml_master.agents import model_advisor

df = pd.read_csv("../resources/data.csv")

ret = model_advisor.invoke(
    df, 
    target_column="Price", # predicting the price of the car
    metric_name="R2", # using R2 to eval
)

print("Recommendation from AI:")
print(ret)
