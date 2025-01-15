import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Simulated dataset of exoplanets with features that might determine habitability
np.random.seed(42)
data = {
    'distance_to_star': np.random.uniform(0.1, 3, 1000),  # in AU
    'star_mass': np.random.uniform(0.5, 1.5, 1000),  # in solar masses
    'star_temperature': np.random.uniform(3000, 7000, 1000),  # in Kelvin
    'planet_radius': np.random.uniform(0.5, 2, 1000),  # in Earth radii
    'orbital_period': np.random.uniform(10, 1000, 1000),  # in Earth days
    'habitable': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # 0: Not habitable, 1: Potentially habitable
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('habitable', axis=1)
y = df['habitable']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict if a new exoplanet might be habitable
def predict_habitability(distance_to_star, star_mass, star_temperature, planet_radius, orbital_period):
    new_exoplanet = np.array([[distance_to_star, star_mass, star_temperature, planet_radius, orbital_period]])
    new_exoplanet_scaled = scaler.transform(new_exoplanet)
    prediction = rf_classifier.predict(new_exoplanet_scaled)
    return "Potentially Habitable" if prediction[0] == 1 else "Not Habitable"

# Example prediction
example_exoplanet = predict_habitability(1.0, 1.0, 5778, 1.0, 365)  # Earth-like parameters
print(f"\nExample Prediction: {example_exoplanet}")
