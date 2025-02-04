import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
data_path = 'neurotransmitter_drug_data.csv'  # Ensure this file exists
if not os.path.exists(data_path):
    st.error("Dataset not found. Please upload 'neurotransmitter_drug_data.csv'.")
    st.stop()

df = pd.read_csv(data_path)

# Define features and target variables
X = df[['Drug Type', 'Dose (mg)', 'Frequency (per week)', 'Duration (months)']]
y = df[['Dopamine Level (ng/mL)', 'Serotonin Level (ng/mL)', 'GABA Level (ng/mL)', 'Glutamate Level (ng/mL)']]

# Preprocessing pipeline
categorical_features = ['Drug Type']
numerical_features = ['Dose (mg)', 'Frequency (per week)', 'Duration (months)']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Save model
model_filename = 'neurotransmitter_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

# Streamlit app
def predict_neurotransmitter_levels(drug_type, dose, frequency, duration):
    input_data = pd.DataFrame({
        'Drug Type': [drug_type],
        'Dose (mg)': [dose],
        'Frequency (per week)': [frequency],
        'Duration (months)': [duration]
    })
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.image("drugs.jpg")
    st.title("Neurotransmitter Level Prediction")
    st.markdown("Enter details about drug exposure to predict neurotransmitter levels.")
    
    # User inputs
    drug_type = st.selectbox("Select Drug Type", df['Drug Type'].unique())
    dose = st.slider("Dose (mg)", 5, 500, 50)
    frequency = st.slider("Frequency (per week)", 1, 7, 3)
    duration = st.slider("Duration (months)", 1, 60, 12)
    
    if st.button("Predict"):
        dopamine, serotonin, gaba, glutamate = predict_neurotransmitter_levels(drug_type, dose, frequency, duration)
        
        st.subheader("Predicted Neurotransmitter Levels")
        st.write(f"Dopamine: {dopamine:.2f} ng/mL")
        st.write(f"Serotonin: {serotonin:.2f} ng/mL")
        st.write(f"GABA: {gaba:.2f} ng/mL")
        st.write(f"Glutamate: {glutamate:.2f} ng/mL")

if __name__ == "__main__":
    main()
