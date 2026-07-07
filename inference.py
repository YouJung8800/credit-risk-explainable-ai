import torch
from credit_risk_model import CreditRiskPipeline

class CreditRiskInference:
    def __init__(self, model_path):
        self.pipeline = CreditRiskPipeline(input_dim=5)
    
    def predict(self, new_customer_data):
        return self.pipeline.predict_proba(new_customer_data)
