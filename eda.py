import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Charger les données
df = pd.read_csv("logs.csv")

# Apercu des 5 premières lignes
print(df.head())

# Information sur les colonnes
print("\n Infos sur les données : ")
print(df.info())

#Statistique descriptives
print("\n Statistiques : ")
print(df.describe())

#Nombre de redémarrages
print("\n Nombre de redémarrages (0 = non, 1 = oui) :")
print(df["rebooted"].value_counts())

#Corrélation entre les variables
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=["timestamp"]).corr(), annot=True, cmap="coolwarm")
plt.title("🔗 Matrice de corrélation")
plt.show()

#Histogramme de l'utilisation CPU
plt.figure(figsize=(6, 4))
sns.histplot(df["cpu_usage"], bins=30, kde=True)
plt.title("Répartition de l'utilisation CPU")
plt.xlabel("CPU Usage (%)")
plt.ylabel("Fréquence")
plt.show()

#Reboot vs. CPU usage
plt.figure(figsize=(6, 4))
sns.boxplot(x="rebooted", y="cpu_usage", data=df)
plt.title("💥 CPU Usage selon le redémarrage")
plt.xlabel("Redémarrage")
plt.ylabel("CPU Usage")
plt.show()