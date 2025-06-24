import logging
import pandas as pd
import time
from ml_master.workflow import solve
#  DEEPSEEK_API_KEY='sk-f91476d2d0b54831947ef8123cc52c83' python workflow.py
logging.basicConfig(level=logging.INFO)


# # Example: Predicting the price of a house using a dataset
# DATA_NAME = "./resources/data.csv"
# target_col = "Price"

# # Example: Predicting the probability of getting lung canncer
# DATA_NAME = "./resources/cancer patient data sets.csv"
# target_col = "chronic Lung Disease"

# DATA_NAME = "./resources/house_prices_train.csv"
# target_col = "SalePrice"

# # Example: Predicting the probability of getting lung canncer
# DATA_NAME = "./resources/WA_Marketing-Campaign.csv"
# target_col = "SalesInThousands"

# Example: Classification task
DATA_NAME = "./resources/creditcard.csv"
target_col = "Class"

# # Example: Prediction task
# DATA_NAME = "./resources/Bank_Personal_Loan_Modelling.csv"
# target_col = "Personal Loan"

# # Example: https://www.kaggle.com/datasets/tristan581/17k-apple-app-store-strategy-games
# DATA_NAME = "./resources/appstore_games.csv"
# target_col = "User Rating Count"

# # # Example: https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability
# DATA_NAME = "./resources/water_potability.csv"
# target_col = "Potability"

# Example: https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset
# DATA_NAME = "./resources/cybersecurity_intrusion_data.csv"
# target_col = "attack_detected"

# # Example: https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset
# DATA_NAME = "./resources/Data_Tanaman_Padi_Sumatera_version_1.csv"
# target_col = "Produksi"


df = pd.read_csv(DATA_NAME)
start_time = time.time()
code_impl, ret_str = solve(
    data_file_name=DATA_NAME,
    target_col=target_col, # predicting the price of the car
    metric_name="R2", # using R2 to eval
)

print("```python")
print(code_impl)
print("```")

print("## Output")
print("```text")
print(ret_str)
print("```")
print('time used:', time.time() - start_time)
