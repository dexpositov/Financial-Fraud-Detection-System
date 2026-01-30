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
        self.customer_profiles = None  # To store customer profiles (the fit method will do it)
        self.defaults = {} # To store default values for new customers

    def fit(self, X, y=None):
        '''
        Fit method to extract the profile of customers and compute default values for new customers.'''
        # Reset customer profiles and defaults
        self.customer_profiles = None
        self.defaults = {}

        # Extract customer profiles from the transaction data
        self.customer_profiles = self.txn_to_customer_features(X, self.kmeans_pipeline)

        # Predict cluster IDs and distances
        cluster_ids = self.kmeans_pipeline.predict(self.customer_profiles)
        centroid_distances = self.kmeans_pipeline.transform(self.customer_profiles)

        self.customer_profiles['Cluster_ID'] = cluster_ids
        for i in range(centroid_distances.shape[1]):
            self.customer_profiles[f'Distance_to_Centroid_{i}'] = centroid_distances[:, i]

        # Compute the default parameters for new customers based on Standard customer profile (Cluster 2)
        default_mask = (self.customer_profiles['Cluster_ID'] == 2)
        if not default_mask.any():
            print("⚠️ Using overall mean for default customer profile as no Standard customers were found in the data.")
            default_mask = slice(None) # Select all rows

        self.defaults = {col: self.customer_profiles.loc[default_mask, col].mean() for col in self.kmeans_pipeline.named_steps['scaler'].feature_names_in_}
        
        self.defaults['Cluster_ID'] = 2  # Default cluster is Standard Customers

        centroids = self.kmeans_pipeline.named_steps['kmeans'].cluster_centers_
        c2_center = centroids[2]

        # Store default distances to centroids for new customers
        for i in range(centroid_distances.shape[1]):
            self.defaults[f'Distance_to_Centroid_{i}'] = np.linalg.norm(c2_center - centroids[i])

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
        df['Is_Night'] = np.where((df['Hour'] >= 2) & (df['Hour'] < 8), 1, 0)
        
        # CLUSTERING FEATURES
        # Copy customer profiles to avoid modifying the original during transform
        df_clients = self.customer_profiles.copy()
        # Select relevant columns to merge
        cols_to_merge = ['Avg_Ticket', 'Std_Ticket', 'Cluster_ID']
        # Add distance to centroid columns
        n_clusters = self.kmeans_pipeline.named_steps['kmeans'].cluster_centers_.shape[0]
        dist_cols = [f'Distance_to_Centroid_{i}' for i in range(n_clusters)]
        cols_to_merge.extend(dist_cols)
 
        # Use a left join to add the client features to transactions
        df = df.merge(df_clients[cols_to_merge], left_on='Customer_ID', right_index=True, how='left', suffixes=('', '_drop'))
        # Drop the extra columns created by the merge (there should be no overlap, but just to be safe)
        df.drop([col for col in df.columns if '_drop' in col], axis=1, inplace=True)
        # For new customers (NaNs), fill with default values
        df.fillna(self.defaults, inplace=True)
        # Ensure Cluster_ID is integer for the One-Hot Encoding later
        df['Cluster_ID'] = df['Cluster_ID'].astype(int)

        # AMOUNT FEATURES
        # Average Ticket per Customer is already merged from customer profiles
        # MAGNITUDE RELATED FEATURES (Amount Ratio)
        df['Amount_Ratio'] = df['Amount'] / df['Avg_Ticket']
        
        # VELOCITY RELATED FEATURES (Time Since Last)
        # Sort by Customer_ID and Timestamp, then group by Customer_ID and calculate time difference in seconds
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

        # LOCATION FEATURES
        # Flag if the transaction location is different from the customer's home location
        df['Is_Foreign'] = np.where(df['Customer_Home'] != df['Location'], 1, 0)
        
        # FINAL FEATURE SELECTION AND RETURNING THE DATAFRAME
        # Select relevant features for the model (One-Hot Encoding is done later in the pipeline)
        df = df.sort_index()  # Restore original order

        base_features = [
            'Amount', 'Amount_Ratio', 'Hour', 'Category', 'Is_Night', 
            'Is_Fixed', 'Time_Since_Last', 'Transactions_Last_Hour', 
            'Is_Foreign', 'Avg_Ticket', 'Std_Ticket', 'Cluster_ID'
        ]

        final_features = base_features + dist_cols

        if df.isnull().any().any():
            raise ValueError("⚠️ NaN values found in the transformed dataframe. Please check the preprocessing steps.")

        return df[final_features]
    
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
        customer_data = customer_features.join(cat_pct, how='left').fillna(0) # Fill NaNs just to make it robust (our generated data does not have gaps when doing the left join)

        # Ensure columns are in the exact same order as training
        # We can get the feature names from the scaler inside the pipeline
        required_features = kmeans_pipeline.named_steps['scaler'].feature_names_in_
        customer_data = customer_data[required_features]

        return customer_data