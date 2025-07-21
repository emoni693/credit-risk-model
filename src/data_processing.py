import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()

        # Convert to datetime
        df['transactionstarttime'] = pd.to_datetime(df['transactionstarttime'])

        # RFM Feature Engineering
        rfm = df.groupby('customerid').agg(
            recency=('transactionstarttime', lambda x: (x.max() - x.min()).days),
            frequency=('transactionid', 'count'),
            monetary=('amount', 'sum')  # You can change to 'value' if preferred
        ).reset_index()

        return rfm


def build_pipeline():
    return DataProcessor()
