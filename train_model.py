import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt 
import joblib
# Charger les données
df = pd.read_csv("logs.csv")

# Supprimer la colonne inutile
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# Séparer features et cible
X = df.drop(columns=["rebooted"])
y = df["rebooted"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🧪 Appliquer SMOTE sur les données d'entraînement
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Créer et entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)
print("✅ Modèle entraîné avec données équilibrées !")

# Prédiction et évaluation
y_pred = model.predict(X_test_scaled)

print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred))

print("=== Matrice de confusion ===")
print(confusion_matrix(y_test, y_pred))

print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")


# Récupérer l’importance des features
importances = model.feature_importances_
feature_names = X.columns

# Trier par importance décroissante
sorted_indices = importances.argsort()[::-1]

# Afficher
plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[sorted_indices], align="center")
plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
plt.title("Importance des variables")
plt.tight_layout()
plt.show()

import joblib

# Sauvegarder le modèle
joblib.dump(model, "random_forest_model.pkl")

# Sauvegarder le scaler aussi
joblib.dump(scaler, "scaler.pkl")

print("✅ Modèle et scaler sauvegardés avec succès.")

import matplotlib.pyplot as plt
import seaborn as sns

# Matrice de confusion visuelle
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Pas de panne", "Panne"],
            yticklabels=["Pas de panne", "Panne"])
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.title("Matrice de confusion")
plt.tight_layout()
plt.show()