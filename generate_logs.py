import pandas as pd
import random

# Nombre de logs
num_logs = 20

# Fonction pour générer un log
def generate_log():
    cpu = random.uniform(0, 100)
    ram = random.uniform(0, 100)
    disk_errors = random.randint(0, 10)
    uptime = random.uniform(0, 500)

    # Logique de détection
    failure = 1 if cpu > 85 or ram > 90 or disk_errors > 5 else 0
    return [cpu, ram, disk_errors, uptime, failure]

# Générer les logs
logs = [generate_log() for _ in range(num_logs)]

# Création d'un DataFrame
df = pd.DataFrame(logs, columns=["CPU_Usage", "RAM_Usage", "Disk_Errors", "Uptime", "Failure"])

# Sauvegarde dans un fichier CSV
df.to_csv("logs.csv", index=False)

print("Fichier 'logs.csv' généré avec succès !")
