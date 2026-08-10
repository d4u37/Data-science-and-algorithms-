"""
Step 1: Load data, run EDA, clean & preprocess.
Dataset: UCI "Default of Credit Card Clients"
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("data.csv")
df.rename(columns={"default.payment.next.month": "DEFAULT"}, inplace=True)
df.drop(columns=["ID"], inplace=True)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().sum(), "total missing")
print("\nTarget distribution:\n", df["DEFAULT"].value_counts(normalize=True))

# ---------------------------------------------------------
# 2. Clean known messy categories
#    EDUCATION: 0,5,6 -> "others" (4)
#    MARRIAGE: 0 -> "others" (3)
# ---------------------------------------------------------
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# ---------------------------------------------------------
# 3. EDA plots
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# target balance
df["DEFAULT"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["#4C72B0", "#DD8452"])
axes[0, 0].set_title("Default Distribution (0 = No, 1 = Yes)")
axes[0, 0].set_xticklabels(["No Default", "Default"], rotation=0)

# age distribution by default
sns.histplot(data=df, x="AGE", hue="DEFAULT", multiple="stack", bins=30, ax=axes[0, 1])
axes[0, 1].set_title("Age Distribution by Default Status")

# credit limit by default
sns.boxplot(data=df, x="DEFAULT", y="LIMIT_BAL", ax=axes[1, 0])
axes[1, 0].set_title("Credit Limit vs Default")
axes[1, 0].set_xticklabels(["No Default", "Default"])

# correlation of PAY_0 (most recent repayment status) with default
pay_default = df.groupby("PAY_0")["DEFAULT"].mean()
pay_default.plot(kind="bar", ax=axes[1, 1], color="#55A868")
axes[1, 1].set_title("Default Rate by Recent Repayment Status (PAY_0)")
axes[1, 1].set_xlabel("PAY_0 (-1=pay duly, 1+=months late)")
axes[1, 1].set_ylabel("Default Rate")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120)
print("\nSaved EDA plot -> eda_overview.png")

# ---------------------------------------------------------
# 4. Save cleaned data
# ---------------------------------------------------------
df.to_csv("data_cleaned.csv", index=False)
print("\nSaved cleaned data -> data_cleaned.csv")
print(df.head())
