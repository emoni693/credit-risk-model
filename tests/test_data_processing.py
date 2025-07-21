import unittest
import pandas as pd
from src.data_processing import build_pipeline

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Sample raw data for testing
        self.raw_df = pd.DataFrame({
            'transactionid': [1, 2],
            'batchid': [101, 102],
            'accountid': ['A1', 'A2'],
            'subscriptionid': ['S1', 'S2'],
            'customerid': ['CustomerId_1', 'CustomerId_2'],
            'currencycode': ['USD', 'USD'],
            'countrycode': ['US', 'US'],
            'providerid': ['P1', 'P2'],
            'productid': ['Prod1', 'Prod2'],
            'productcategory': ['Cat1', 'Cat2'],
            'channelid': ['C1', 'C2'],
            'amount': [100, 200],
            'value': [10000, 20000],
            'transactionstarttime': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'pricingstrategy': [1, 2],
            'fraudresult': [0, 1]
        })

    def test_pipeline_runs(self):
        pipeline = build_pipeline()
        result_df = pipeline.fit_transform(self.raw_df)
        self.assertIsInstance(result_df, pd.DataFrame)

    def test_pipeline_adds_columns(self):
        pipeline = build_pipeline()
        result_df = pipeline.fit_transform(self.raw_df)
        expected_columns = {'recency', 'frequency', 'monetary'}
        self.assertTrue(expected_columns.issubset(result_df.columns))

if __name__ == '__main__':
    unittest.main()
