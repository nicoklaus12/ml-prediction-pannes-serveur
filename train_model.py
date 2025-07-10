import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import joblib

# Charger les données
df = pd.read_csv("logs.csv")

# Séparer les features (X) et la cible (y)
X = df.drop("Failure", axis=1)
y = df["Failure"]

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédire sur le test
y_pred = model.predict(X_test)

# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Sauvegarder le modèle et scaler
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modèle et scaler sauvegardés avec succès.")

joblib.dump(model, "model.pkl")
print("✅Modèle sauvegardé dans model.pkl!!")