# src/run_pipeline.py
import pandas as pd
from data_processing import build_pipeline  # Remove `src.`

raw_df = pd.read_csv("../data/data.csv")  # Go one directory up
pipeline = build_pipeline()
processed_df = pipeline.fit_transform(raw_df)

print(processed_df.head())
