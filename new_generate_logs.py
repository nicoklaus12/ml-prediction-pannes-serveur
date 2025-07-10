import pandas as pd
import random

# Nombre de lignes à générer
nb_lignes = 20

# Génération de données simulées
data = {
    "CPU_Usage": [random.uniform(0, 100) for _ in range(nb_lignes)],
    "RAM_Usage": [random.uniform(0, 100) for _ in range(nb_lignes)],
    "Disk_Errors": [random.uniform(0, 500) for _ in range(nb_lignes)],
    "Uptime": [random.uniform(0, 1000) for _ in range(nb_lignes)],
     "Failure": [random.uniform(0, 1000) for _ in range(nb_lignes)],


    
}

# Créer un DataFrame
df = pd.DataFrame(data)

# Sauvegarder dans un fichier CSV
df.to_csv("nouveaux_logs.csv", index=False)
print(f"✅ {nb_lignes} lignes générées dans 'nouveaux_logs.csv'")