import pandas as pd
import joblib
import sys

# 1. Vérifier si le fichier de logs a été fourni
if len(sys.argv) < 2:
    print("Utilisation : python predict.py <chemin_fichier_logs.csv>")
    sys.exit(1)

file_path = sys.argv[1]





# 2. Charger le modèle
try:
    model = joblib.load("model.pkl")
    print("✅ Modèle chargé avec succès.")
except:
    print("❌ Erreur : le fichier model.pkl est introuvable.")
    sys.exit(1)

# 3. Charger le scaler
try:
    scaler = joblib.load("scaler.pkl")
    print("✅ Scaler chargé avec succès.")
except:
    print("❌ Erreur : le fichier scaler.pkl est introuvable.")
    sys.exit(1)

# 4. Charger le fichier de logs à prédire
try:
    data = pd.read_csv(file_path)
    
    # Supprimer la colonne "Failure" si elle est présente
    if "Failure" in data.columns:
        data = data.drop("Failure", axis=1)
    
    print(f"✅ Fichier '{file_path}' chargé. {len(data)} lignes à prédire.")
except:
    print(f"❌ Erreur lors du chargement du fichier '{file_path}'")
    sys.exit(1)


# 5. Appliquer le scaler et faire les prédictions
try:
    # Normaliser les données comme à l'entraînement
    data_scaled = scaler.transform(data)

    predictions = model.predict(data_scaled)

    print("\n🔎 Résultats des prédictions :")
    for i, pred in enumerate(predictions):
        status = "❌ Panne probable" if pred == 1 else "✅ Pas de panne"
        print(f"Ligne {i+1}: {status}")

except Exception as e:
    print("❌ Erreur lors de la prédiction :", e)


