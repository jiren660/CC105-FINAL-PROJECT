import pickle
import os

MODELS_PATH = r'c:\Users\Admin\jupyter\sleep_health_project\predictor\models'
encoders = pickle.load(open(os.path.join(MODELS_PATH, 'encoders.pkl'), 'rb'))

print("BMI Category classes:", encoders['BMI Category'].classes_)
print("All encoders keys:", encoders.keys())
