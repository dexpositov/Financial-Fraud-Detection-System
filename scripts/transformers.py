
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, kmeans_pipeline):
        """
        kmeans_pipeline: Pre-trained clustering pipeline (.pkl)
        transaction_df: Transaction dataframe to preprocess 
        """
        self.kmeans_pipeline = kmeans_pipeline

    def fit(self, X, y=None):
        # Our k-means pipline is pre-trained, so we do not need to fit anything here.
        return self

    def transform(self, X):
        # We do not modify the original dataframe
        df = X.copy()

        # DATETIME PARSING
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # TIME FEATURES (Hour, Is_Night)
        # Extract hour from timestamp
        df['Hour'] = df['Timestamp'].dt.hour
        
        # Is_Night feature: 1 if transaction is between 2 AM and 8 AM
        df['Is_Night'] = df['Hour'].apply(lambda x: 1 if 2 <= x < 8 else 0)
        
        # CLUSTERING FEATURES
        # Use the static method to extract customer features
        df_clients = self.txn_to_customer_features(df, self.kmeans_pipeline)

        # Predict cluster IDs and distances
        cluster_ids = self.kmeans_pipeline.predict(df_clients)
        centroid_distances = self.kmeans_pipeline.named_steps['kmeans'].transform(df_clients)
 
        # Map distances and IDs back to the original dataframe
        for i in range(centroid_distances.shape[1]):
            df[f'Distance_to_Centroid_{i}'] = df["Customer_ID"].map(dict(zip(df_clients.index, centroid_distances[:, i])))

        df['Cluster_ID'] = df["Customer_ID"].map(dict(zip(df_clients.index, cluster_ids)))

        # AMOUNT FEATURES
        # Average Ticket per Customer
        df['Avg_Ticket'] = df['Customer_ID'].map(dict(zip(df_clients.index, df_clients['Avg_Ticket'])))
        
        # MAGNITUDE RELATED FEATURES (Amount Ratio)
        df['Amount_Ratio'] = df['Amount'] / df['Avg_Ticket']
        
        # VELOCITY RELATED FEATURES (Time Since Last)
        # Sort by Customer_ID and Timestamp (Customer ID is not needed to be sorted, but for clarity), then group by Customer_ID and calculate time difference in seconds
        
        df['Time_Since_Last'] = df.sort_values(['Customer_ID', 'Timestamp']).groupby('Customer_ID')['Timestamp'].diff().dt.total_seconds()

        # Fill NaN values (first transaction for each customer) with the average time difference for that customer
        cust_avg_time_diff = df.groupby('Customer_ID')['Time_Since_Last'].transform('mean')
        df['Time_Since_Last'] = df['Time_Since_Last'].fillna(cust_avg_time_diff)

        # Just to make the code robust, fill any remaining NaNs with overall mean
        df['Time_Since_Last'] = df['Time_Since_Last'].fillna(df['Time_Since_Last'].mean())

        # Now, we compute the number of transaction made by the customer in the last hour. We use rolling window for that.
        # First, we sort the dataframe by Customer_ID and Timestamp
        df = df.sort_values(['Customer_ID', 'Timestamp'])
        hour_counts = df.set_index('Timestamp').groupby('Customer_ID')['Transaction_ID'].rolling('1H').count() - 1
        df['Transactions_Last_Hour'] = hour_counts.values
        df = df.sort_index()  # Restore original order

        # LOCATION FEATURES
        # Flag if the transaction location is different from the customer's home location
        df['Is_Foreign'] = df.apply(lambda x: 1 if x['Customer_Home'] != x['Location'] else 0, axis=1)
        
        # FINAL CLEANUP
        # Select relevant features for the model (One-Hot Encoding is done later in the pipeline)

        features_finales = [
            'Amount', 'Amount_Ratio', 'Hour', 'Category', 'Is_Night', 'Is_Fixed',
            'Time_Since_Last', 'Transactions_Last_Hour', 'Is_Foreign', 'Avg_Ticket',
            'Cluster_ID', 'Distance_to_Centroid_0', 'Distance_to_Centroid_1', 
            'Distance_to_Centroid_2', 'Distance_to_Centroid_3'
        ]
        
        return df[features_finales]
    
    @staticmethod
    def txn_to_customer_features(transaction_df, kmeans_pipeline):
        """
        Extract customer-level features from transaction data.
        
        Parameters:
        transaction_df (pd.DataFrame): DataFrame containing transaction data with columns:
                        ['Customer_ID', 'Transaction_ID', 'Amount', 'Category']
        kmeans_pipeline: The fitted KMeans pipeline used for customer segmentation.

        Returns:
        pd.DataFrame: DataFrame with customer-level features compatible with the customer segmentation K-means model.
        """
        # Filter out fraudulent transactions if 'Is_Fraud' column exists
        if 'Is_Fraud' in transaction_df.columns:
            clean_txn_df = transaction_df[transaction_df['Is_Fraud'] == 0]
        else:
            clean_txn_df = transaction_df

        # Basic Customer Features
        customer_features = clean_txn_df.groupby('Customer_ID').agg(
            Total_Spend=('Amount', 'sum'),
            Transaction_Count=('Transaction_ID', 'count'),
            Avg_Ticket=('Amount', 'mean'),
            Std_Ticket=('Amount', 'std'),
            Max_Ticket=('Amount', 'max'),
            Min_Ticket=('Amount', 'min')
        )

        # Category Percentages
        cat_pivot = clean_txn_df.pivot_table(
            index='Customer_ID', 
            columns='Category', 
            values='Amount', 
            aggfunc='sum', 
            fill_value=0
        )
        # Convert to percentages
        cat_pct = cat_pivot.div(cat_pivot.sum(axis=1), axis=0)
        cat_pct.columns = [f'Pct_{c}' for c in cat_pct.columns]

        # Join to create the input vector X_customers
        customer_data = customer_features.join(cat_pct, how='left').fillna(0) # Fill NaNs just to make it robust (our generated data does not have NaNs)

        # Ensure columns are in the exact same order as training
        # We can get the feature names from the scaler inside the pipeline
        required_features = kmeans_pipeline.named_steps['scaler'].feature_names_in_
        customer_data = customer_data[required_features]

        return customer_data