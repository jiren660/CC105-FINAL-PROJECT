from django.shortcuts import render
import pickle
import pandas as pd
import os

# Kunin ang path ng models folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(CURRENT_DIR, 'models')

# I-load ang mga model files
model = pickle.load(open(os.path.join(MODELS_PATH, 'sleep_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'rb'))
encoders = pickle.load(open(os.path.join(MODELS_PATH, 'encoders.pkl'), 'rb'))
feature_names = pickle.load(open(os.path.join(MODELS_PATH, 'feature_names.pkl'), 'rb'))

def home(request):
    # Base list from encoder
    original_classes = list(encoders['BMI Category'].classes_)
    
    # We only want 'Normal' and a combined 'Overweight/Obese'
    bmi_display_list = []
    if 'Normal' in original_classes:
        bmi_display_list.append('Normal')
    
    # Combine Overweight and Obese into one UI option
    if 'Overweight' in original_classes or 'Obese' in original_classes:
        bmi_display_list.append('Overweight/Obese')
            
    context = {
        'occupations': sorted(encoders['Occupation'].classes_),
        'genders': encoders['Gender'].classes_,
        'bmi_categories': sorted(bmi_display_list)
    }
    return render(request, 'predict.html', context)

def predict(request):
    if request.method == 'POST':
        data = request.POST
        
        # 1. Feature Engineering
        duration = float(data['duration'])
        quality = float(data['quality'])
        steps = float(data['steps'])
        activity = float(data['activity'])
        
        sleep_efficiency = quality * duration
        physical_intensity = activity / (steps + 1)

        # 2. Encoding Function
        def safe_encode(encoder_name, value):
            le = encoders[encoder_name]
            if value not in le.classes_:
                return le.transform([le.classes_[0]])[0]
            return le.transform([value])[0]

        gender_enc = safe_encode('Gender', data['gender'])
        occ_enc = safe_encode('Occupation', data['occupation'])
        
        # Map 'Overweight/Obese' back to 'Overweight' for the model
        bmi_input = data['bmi']
        if bmi_input == 'Overweight/Obese':
            bmi_input = 'Overweight'
            
        bmi_enc = safe_encode('BMI Category', bmi_input)

        # 3. Create the input dictionary with ALL columns
        input_dict = {
            'Gender': gender_enc,
            'Age': float(data['age']),
            'Occupation': occ_enc,
            'Sleep Duration': duration,
            'Quality of Sleep': quality,
            'Physical Activity Level': activity,
            'Stress Level': float(data['stress']),
            'BMI Category': bmi_enc,
            'Heart Rate': float(data['heart_rate']),
            'Daily Steps': steps,
            'Systolic': int(data['systolic']),
            'Diastolic': int(data['diastolic']),
            'Sleep Efficiency': sleep_efficiency,
            'Physical_Intensity': physical_intensity
        }
        
        # 4. Convert to DataFrame and FIX COLUMN ORDER FIRST
        # The dataframe MUST have the exact same column order as X_train
        df_final = pd.DataFrame([input_dict])
        df_final = df_final[feature_names] 

        # 5. SCALE EVERYTHING (Since the scaler was fit on all 14 columns)
        # We transform the whole dataframe because that's what we did in Jupyter
        df_scaled = scaler.transform(df_final)

        # 6. Prediction
        prediction_num = model.predict(df_scaled)[0]
        result = encoders['Sleep Disorder'].inverse_transform([prediction_num])[0]

        # 7. Identify Positive Indicators (for the UI)
        indicators = []
        stress_val = float(data['stress'])
        if stress_val <= 4:
            indicators.append({'label': 'Low Stress', 'icon': 'fa-smile-beam', 'class': 'text-success'})
        elif stress_val <= 7:
            indicators.append({'label': 'Manageable Stress', 'icon': 'fa-meh', 'class': 'text-warning'})
            
        systolic = int(data['systolic'])
        diastolic = int(data['diastolic'])
        if systolic <= 120 and diastolic <= 80:
            indicators.append({'label': 'Optimal BP', 'icon': 'fa-heart', 'class': 'text-success'})
        elif systolic <= 130 and diastolic <= 85:
            indicators.append({'label': 'Normal BP', 'icon': 'fa-heart', 'class': 'text-primary'})

        context = {
            'result': result,
            'efficiency': min(round(sleep_efficiency, 1), 100.0),
            'user_data': data,
            'intensity': round(physical_intensity, 4),
            'indicators': indicators
        }

        return render(request, 'result.html', context)