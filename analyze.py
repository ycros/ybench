import pandas as pd
from prettytable import PrettyTable

# Load data
df = pd.read_json("test.jsonl", lines=True)

# Calculate tokens per second
df["tps"] = (df["prompt_tokens"] + df["gen_tokens"]) / df["duration"]


# Function to calculate metrics
def calculate_metrics(dataframe):
    avg_tps = dataframe["tps"].mean().round(2)
    percentile_99_tps = dataframe["tps"].quantile(0.99).round(2)
    median_tps = dataframe["tps"].median().round(2)
    std_dev_tps = dataframe["tps"].std().round(2)
    return avg_tps, percentile_99_tps, median_tps, std_dev_tps


# Initialize PrettyTable
table = PrettyTable()
table.field_names = [
    "Category",
    "Average TPS",
    "99th Pct TPS",
    "Median TPS",
    "Std Dev TPS",
]
table.float_format = "0.1"
table.align = "r"

# Overall metrics
overall_metrics = calculate_metrics(df)
table.add_row(["Overall", *overall_metrics])

# Calculate quartiles for breakdowns
gen_tokens_75th, prompt_tokens_25th = df["gen_tokens"].quantile(0.75), df[
    "prompt_tokens"
].quantile(0.25)
prompt_tokens_75th, gen_tokens_25th = df["prompt_tokens"].quantile(0.75), df[
    "gen_tokens"
].quantile(0.25)

# Balanced case
gen_tokens_iqr, prompt_tokens_iqr = df["gen_tokens"].between(
    gen_tokens_25th, gen_tokens_75th
), df["prompt_tokens"].between(prompt_tokens_25th, prompt_tokens_75th)
balanced_case = df[gen_tokens_iqr & prompt_tokens_iqr]
balanced_case_metrics = calculate_metrics(balanced_case)
table.add_row(["Balanced Case", *balanced_case_metrics])

# High gen_tokens with low prompt_tokens
high_gen_low_prompt = df[
    (df["gen_tokens"] >= gen_tokens_75th) & (df["prompt_tokens"] <= prompt_tokens_25th)
]
high_gen_low_prompt_metrics = calculate_metrics(high_gen_low_prompt)
table.add_row(["High Gen / Low Prompt", *high_gen_low_prompt_metrics])

# High prompt_tokens with low gen_tokens
high_prompt_low_gen = df[
    (df["prompt_tokens"] >= prompt_tokens_75th) & (df["gen_tokens"] <= gen_tokens_25th)
]
high_prompt_low_gen_metrics = calculate_metrics(high_prompt_low_gen)
table.add_row(["High Prompt / Low Gen", *high_prompt_low_gen_metrics])

# Print the table
print(table)
