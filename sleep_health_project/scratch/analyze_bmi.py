import pandas as pd

# Load CSV and treat 'None' as a string, not NaN
df = pd.read_csv(r'c:\Users\Admin\jupyter\dataset\Sleep_health_and_lifestyle_dataset.csv').fillna('None')

print("--- BMI Category Distribution ---")
print(df['BMI Category'].value_counts())

print("\n--- Sleep Disorder by BMI Category (Counts) ---")
ct = pd.crosstab(df['BMI Category'], df['Sleep Disorder'])
print(ct)

print("\n--- Percentage Distribution (%) ---")
ct_pct = pd.crosstab(df['BMI Category'], df['Sleep Disorder'], normalize='index') * 100
print(ct_pct)
