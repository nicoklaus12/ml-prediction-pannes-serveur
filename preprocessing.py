import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger les données
df = pd.read_csv("logs.csv")

# Vérifier si 'timestamp' existe, et le supprimer si c'est le cas
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# Séparer les features (X) et la cible (y)
X = df.drop(columns=["rebooted"])
y = df["rebooted"]

# Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Données préparées avec succès.")
