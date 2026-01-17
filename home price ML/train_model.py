import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
df = pd.read_csv("sample_house_data.csv")

X = df.drop("price", axis=1)
y = df["price"]

# Categorical & numerical split
categorical_cols = ["location"]
numeric_cols = ["area", "bedrooms", "bathrooms"]

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train
pipeline.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully.")
