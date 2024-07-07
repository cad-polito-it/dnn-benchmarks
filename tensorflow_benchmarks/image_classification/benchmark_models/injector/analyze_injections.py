# %%

import pandas as pd


# %%

FILE_NAME = "../../reports/CIFAR10/ResNet20/CIFAR10_ResNet20_240208_1627.csv"

exp_data = pd.read_csv(FILE_NAME)


# %%
faulty_data = exp_data[1:]

total_injections = faulty_data["n_injections"].sum()

print(total_injections)

# %%

masked_pct = faulty_data["masked"].sum() / total_injections
non_critical = faulty_data["non_critical"].sum() / total_injections
critical = faulty_data["critical"].sum() / total_injections
top_1_correct = faulty_data["top_1_correct"].sum() / total_injections
top_5_correct = faulty_data["top_5_correct"].sum() / total_injections


# %%

result_metrics = faulty_data.sum(numeric_only=True) / total_injections * 100

print(result_metrics)

# %%
