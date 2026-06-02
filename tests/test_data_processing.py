"""
Unit Tests for Data Processing Module
Task 5 Component 6: Unit tests for helper functions
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor


class TestDataProcessor:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        data = pd.DataFrame({
            'CustomerId': np.random.choice(['CUST_A', 'CUST_B', 'CUST_C'], 100),
            'Amount': np.random.uniform(10, 500, 100),
            'Value': np.random.uniform(10, 500, 100),
            'TransactionStartTime': np.random.choice(dates, 100),
            'ProductCategory': np.random.choice(['Electronics', 'Fashion', 'Books'], 100),
            'ChannelId': np.random.choice([1, 2, 3], 100),
            'PricingStrategy': np.random.choice([1, 2], 100)
        })
        return data
    
    def test_create_aggregate_features(self, sample_data):
        """Test that aggregate features are created correctly"""
        processor = DataProcessor()
        agg_features = processor.create_aggregate_features(sample_data)
        
        expected_columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount']
        
        for col in expected_columns:
            assert col in agg_features.columns
        
        assert len(agg_features) == len(sample_data['CustomerId'].unique())
        assert agg_features['TransactionCount'].sum() == len(sample_data)
    
    def test_extract_datetime_features(self, sample_data):
        """Test that datetime features are extracted correctly"""
        processor = DataProcessor()
        result = processor.extract_datetime_features(sample_data)
        
        assert 'TransactionHour' in result.columns
        assert 'TransactionDay' in result.columns
        assert 'TransactionMonth' in result.columns
        assert 'TransactionYear' in result.columns
        
        assert result['TransactionHour'].between(0, 23).all()
        assert result['TransactionDay'].between(1, 31).all()
        assert result['TransactionMonth'].between(1, 12).all()
    
    def test_calculate_rfm_returns_expected_columns(self, sample_data):
        """Test that RFM calculation returns correct columns"""
        processor = DataProcessor()
        rfm = processor.calculate_rfm(sample_data)
        
        assert 'CustomerId' in rfm.columns
        assert 'Monetary' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Recency' in rfm.columns
        
        assert len(rfm) == len(sample_data['CustomerId'].unique())
    
    def test_create_high_risk_proxy_returns_binary_label(self, sample_data):
        """Test that high-risk proxy creates binary is_high_risk column"""
        processor = DataProcessor()
        rfm = processor.calculate_rfm(sample_data)
        result = processor.create_high_risk_proxy(rfm)
        
        assert 'is_high_risk' in result.columns
        assert result['is_high_risk'].isin([0, 1]).all()
        assert result['is_high_risk'].value_counts().sum() == len(result)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])