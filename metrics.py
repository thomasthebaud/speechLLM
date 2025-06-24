import numpy as np

def MAE(target_age,predicted_age):
    if predicted_age=='NA':return 100
    else: return np.abs(float(predicted_age)-float(target_age))
