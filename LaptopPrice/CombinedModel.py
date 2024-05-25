import numpy as np
from joblib import dump, load

class CombinedModel:
    def __init__(self, model1, model2, model3,model4, model5, weights=(0.5, 0.2, 0.15,0.1,0.05)):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.weights = weights
    
    def predict(self, X):
        pred1 = self.model1.predict(X.to_array())
        pred2 = self.model2.predict(X)
        pred3 = self.model3.predict(X)
        pred4 = self.model4.predict(X)
        pred5 = self.model5.predict(X)
        combined_pred = self.weights[0] * pred1 + self.weights[1] * pred2 + self.weights[2] * pred3 + self.weights[3] * pred4 + self.weights[4] * pred5
        return combined_pred
