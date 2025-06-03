import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Nombre de lignes à générer
n = 500

# Générer des timestamps toutes les minutes à partir d'une date de départ
start_time = datetime(2025, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

# Générer des données synthétiques réalistes
np.random.seed(42)
cpu_usage = np.random.normal(loc=50, scale=15, size=n).clip(0, 100)        # % CPU entre 0 et 100
ram_usage = np.random.normal(loc=60, scale=10, size=n).clip(0, 100)        # % RAM entre 0 et 100
disk_io = np.random.normal(loc=200, scale=50, size=n).clip(0)              # I/O disque (unités arbitraires)
error_count = np.random.poisson(lam=1, size=n)                            # nombre d'erreurs
rebooted = np.random.choice([0, 1], size=n, p=[0.9, 0.1])                  # reboot 10% des cas

# Construire le DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "cpu_usage": cpu_usage,
    "ram_usage": ram_usage,
    "disk_io": disk_io,
    "error_count": error_count,
    "rebooted": rebooted
})

# Sauvegarder dans un CSV
df.to_csv("logs.csv", index=False)

print("✅ Fichier logs.csv généré avec", n, "lignes.")
