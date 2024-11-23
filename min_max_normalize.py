import pandas as pd


def min_max_normalize(df):

    normalized_df = pd.DataFrame()

    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()

        normalized_df[column] = (df[column] - min_val) / (max_val - min_val)

    return normalized_df