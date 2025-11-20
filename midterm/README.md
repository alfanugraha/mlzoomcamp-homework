# Midterm ML Zoomcamp - Stroke Prediction

## Study Case Dataset

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately [11% of total deaths](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

[This dataset](https://arxiv.org/pdf/1904.11280.pdf) is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

Attribute Information:

- `id`: unique identifier
- `gender`: “Male”, “Female” or “Other”
- `age`: age of the patient
- `hypertension`: 0 if the patient doesn’t have hypertension, 1 if the patient has hypertension
- `heart_disease`: 0 if the patient doesn’t have any heart diseases, 1 if the patient has a heart disease
- `ever_married`: “No” or “Yes”
- `work_typ`e: “children”, “Govt_jov”, “Never_worked”, “Private” or “Self-employed”
- `Residence_type`: “Rural” or “Urban”
- `avg_glucose_level`: average glucose level in blood
- `bmi`: body mass index
- `smoking_status`: “formerly smoked”, “never smoked”, “smokes” or “Unknown”*
- `stroke`: 1 if the patient had a stroke or 0 if not

## Modelling 

The model used in this project is a classification model that predicts the probability of a patient having a stroke based on the input features. The model was trained using a Logistic Regression, Decision Tree, Random Forest, or Gradient Boosting algorithm. Other potential models have not been explored in this project but could be considered for future work.

## Docker

To use Docker to run the prediction script, follow these steps:

- Download image base `nugrahadocker/stroke-prediction:v0.1` from Docker Hub
- Run container from the image

```bash
docker run --rm nugrahadocker/stroke-prediction:v0.1
```

This will execute the `predict.py` script inside the Docker container and display the stroke prediction probability for the given patient data.

