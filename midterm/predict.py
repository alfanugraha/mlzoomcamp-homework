import pickle

with open('stroke_model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

patient = {'gender': 'Female',
 'age': 72.0,
 'hypertension': 0,
 'heart_disease': 1,
 'ever_married': 'No',
 'residence_type': 'Rural',
 'avg_glucose_level': 124.38,
 'bmi': 23.4,
 'smoking_status': 'formerly smoked',
 'employed': 1 
}

pred = pipeline.predict_proba(patient)[0, 1]
print(pred)