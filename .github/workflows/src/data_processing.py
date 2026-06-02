"""
Feature Engineering Pipeline for Credit Risk Model
Task 3 & 4: Feature Engineering and Proxy Target Variable
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Complete data processing pipeline for credit risk modeling.
    Handles feature engineering, RFM calculation, and proxy target creation.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.numerical_features = ['Amount', 'Value']
        self.categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']
        self.preprocessor = None
        self.kmeans = None
        
    def create_aggregate_features(self, df):
        """
        Create aggregate features per customer
        Task 3 Component 1: Aggregate features
        """
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        
        agg_features.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 
                                'TransactionCount', 'StdAmount']
        agg_features['StdAmount'] = agg_features['StdAmount'].fillna(0)
        
        return agg_features
    
    def extract_datetime_features(self, df):
        """
        Extract datetime features from TransactionStartTime
        Task 3 Component 2: Datetime extraction
        """
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        return df
    
    def calculate_rfm(self, df, snapshot_date=None):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics
        Task 4 Component 1: RFM calculation
        """
        if snapshot_date is None:
            snapshot_date = df['TransactionStartTime'].max()
        
        # Monetary: Total transaction amount per customer
        monetary = df.groupby('CustomerId')['Amount'].sum().reset_index()
        monetary.columns = ['CustomerId', 'Monetary']
        
        # Frequency: Number of transactions per customer
        frequency = df.groupby('CustomerId').size().reset_index()
        frequency.columns = ['CustomerId', 'Frequency']
        
        # Recency: Days since last transaction
        recency = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
        recency.columns = ['CustomerId', 'LastTransactionDate']
        recency['Recency'] = (snapshot_date - recency['LastTransactionDate']).dt.days
        
        # Merge all RFM metrics
        rfm = monetary.merge(frequency, on='CustomerId')
        rfm = rfm.merge(recency[['CustomerId', 'Recency']], on='CustomerId')
        
        return rfm
    
    def create_high_risk_proxy(self, rfm_df):
        """
        Create is_high_risk proxy variable using K-Means clustering
        Task 4 Component 2 & 3: K-Means clustering and high-risk assignment
        """
        # Prepare RFM features for clustering
        rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
        
        # Scale features (important for K-Means)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_features)
        
        # Apply K-Means with k=3
        self.kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(rfm_scaled)
        
        rfm_df['Cluster'] = clusters
        
        # Analyze clusters to identify high-risk (lowest frequency and monetary)
        cluster_stats = rfm_df.groupby('Cluster').agg({
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Recency': 'mean'
        }).round(2)
        
        # High-risk cluster is the one with lowest frequency and monetary
        # (least engaged customers)
        cluster_stats['Score'] = cluster_stats['Frequency'] + cluster_stats['Monetary']
        high_risk_cluster = cluster_stats['Score'].idxmin()
        
        # Assign is_high_risk label
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
        
        return rfm_df
    
    def build_preprocessing_pipeline(self):
        """
        Build sklearn pipeline for feature preprocessing
        Task 3 Component 7: Pipeline structure
        """
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])
        
        return self.preprocessor
    
    def process(self, df):
        """
        Full data processing pipeline
        """
        # Step 1: Extract datetime features
        df = self.extract_datetime_features(df)
        
        # Step 2: Create aggregate features
        agg_features = self.create_aggregate_features(df)
        
        # Step 3: Calculate RFM
        rfm = self.calculate_rfm(df)
        
        # Step 4: Create high-risk proxy
        rfm = self.create_high_risk_proxy(rfm)
        
        # Step 5: Merge target back to features
        result = agg_features.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId')
        
        return result, rfm


if __name__ == "__main__":
    print("Data Processing Module Loaded")
    print("Components included:")
    print("  - Aggregate features (sum, mean, count, std)")
    print("  - Datetime extraction (hour, day, month, year)")
    print("  - Categorical encoding (OneHotEncoder)")
    print("  - Missing value handling (median/mode imputation)")
    print("  - Normalization/Standardization (StandardScaler)")
    print("  - RFM calculation (Recency, Frequency, Monetary)")
    print("  - K-Means clustering for proxy target")
    print("  - is_high_risk binary label creation")