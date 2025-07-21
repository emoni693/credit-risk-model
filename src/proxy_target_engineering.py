import pandas as pd
from datetime import datetime

def calculate_rfm(df, snapshot_date=None):
    import pandas as pd

    # --- 1. Ensure transactionstarttime is datetime and timezone-naive
    df['transactionstarttime'] = pd.to_datetime(df['transactionstarttime'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(df['transactionstarttime']):
        df['transactionstarttime'] = df['transactionstarttime'].dt.tz_convert(None)
    else:
        df['transactionstarttime'] = df['transactionstarttime'].dt.tz_localize(None)

    # --- 2. Ensure snapshot_date is a timezone-naive datetime
    if snapshot_date is None:
        snapshot_date = df['transactionstarttime'].max() + pd.Timedelta(days=1)
    else:
        # FORCE CONVERSION
        snapshot_date = pd.to_datetime(snapshot_date, errors='coerce')

        # FORCE tz-naive by checking type and catching error early
        try:
            if snapshot_date.tzinfo is not None:
                snapshot_date = snapshot_date.tz_convert(None)
        except AttributeError:
            # snapshot_date might still be string if conversion failed
            raise ValueError(f"snapshot_date is not datetime: {snapshot_date}")

    # --- 3. Aggregate RFM
    rfm = df.groupby('customerid').agg({
        'transactionstarttime': lambda x: (snapshot_date - x.max()).days,
        'transactionid': 'count',
        'amount': 'sum'
    }).reset_index()

    rfm.columns = ['customerid', 'recency', 'frequency', 'monetary']
    return rfm

def cluster_customers(rfm_df, n_clusters=3):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df['cluster'] = kmeans.fit_predict(scaled)
    return rfm_df

def assign_high_risk(df):
    df['high_risk'] = df['cluster'].apply(lambda x: 1 if x == df['cluster'].max() else 0)
    return df

# Optional script execution block
if __name__ == "__main__":
    df = pd.read_csv("../data/data.csv")
    df = calculate_rfm(df)
    df = cluster_customers(df)
    df = assign_high_risk(df)
    df.to_csv("../data/labeled_data.csv", index=False)
