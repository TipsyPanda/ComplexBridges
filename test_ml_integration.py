# Test script: test_ml_integration.py
from src.ml_scoring import MLScorer
import pandas as pd
import numpy as np

# Initialize
scorer = MLScorer('artifacts/')

# Create test data
test_data = pd.DataFrame({
    'value': np.random.normal(100, 20, 100),
    'sensor_type': 'strain_gauge',
    'span_id': 'SPAN_1'
})

# Score
result = scorer.compute_risk_score(test_data, 'SPAN_1', 'strain_gauge')
print(result)