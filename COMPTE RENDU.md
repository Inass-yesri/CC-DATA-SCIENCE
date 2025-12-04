## 📄 Détection de Fraude – Compte Rendu du Projet
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/a2c3ebce-f022-4246-843e-c58dcda56ea8" />

Par EL YESRI INASS 
<img width="100" height="150" alt="image" src="https://github.com/user-attachments/assets/8ff73355-eaf0-42d3-ac75-12fdc08df8d2" />

## 1. Introduction

Le fichier DATE.SET.csv constitue la base du projet Machine Learning.

Il contient :

un identifiant client (client_id)

une variable cible continue (target) comprise entre 0 et 1

L’objectif du projet est de développer un modèle prédictif capable d’estimer ce score target pour de nouveaux clients après enrichissement du dataset.

## Ce rapport suit le cahier des charges officiel :

dataset → preprocessing → EDA → modélisation → résultats → conclusion.

## 2. Le Dataset (Livrable 1)

## 2.1. Source & Sélection

Fichier utilisé : DATE.SET.csv

58 069 lignes

2 colonnes

Dataset adapté à un problème réaliste de scoring client, contrairement à des jeux triviaux (Iris, Titanic).

## 2.2. Problématique et type de tâche

## Tâche : Régression supervisée

Objectif : prédire une variable target continue ∈ [0,1]

Application : scoring client, probabilité, intensité, risque.

## 2.3. Dictionnaire de données

Colonne	Type	Rôle	Description

client_id	string	ID Client	Identifiant unique (ex: test_Client_0)

# target	float64	Target	Score continu ∈ [0,1]

## 🔍 Statistiques de base

Min ≈ 0

Max ≈ 1

Moyenne ≈ 0.50

Écart-type ≈ 0.29

## 3. Méthodologie & Graphiques (Livrable 2+3)

## 3.1. Pré-traitement (Preprocessing)
```python
import pandas as pd

df = pd.read_csv("DATE.SET.csv")
df.info()
df.describe()
df.duplicated().sum()
```
## 🎯 Choix techniques justifiés :

Le dataset est propre mais devra être enrichi.

client_id sera utilisé pour joindre d'autres tables.

Aucun modèle n’accepte les strings → encodages nécessaires après jointure.

Normalisation obligatoire si SVM, KNN ou MLP sont utilisés.

## 3.2. Analyse Exploratoire (EDA)

# 📌 Graphique 1 — Histogramme de la variable cible
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7,4))
sns.histplot(df["target"], bins=30, kde=True)
plt.title("Distribution de la variable target")
plt.xlabel("Score target")
plt.ylabel("Fréquence")
plt.show()
```
## Interprétation :

La distribution est quasi uniforme entre 0 et 1, mais une légère densité apparaît autour de 0.5.

Cela confirme :

une bonne variabilité pour la modélisation,

absence de déséquilibre,

pas de transformation de type log à appliquer.

# 📌Graphique 2 — Boxplot de target (détection d’outliers)

```python
plt.figure(figsize=(6,3))
sns.boxplot(x=df["target"])
plt.title("Boxplot de la cible target")
plt.show()
```
## Interprétation :

Le boxplot montre :

aucune valeur aberrante extrême,

une dispersion homogène.

Cela confirme que le dataset ne nécessite pas de traitement d’outliers pour la cible.

# 📌Graphique 3 — Heatmap préliminaire (corrélations)

Ce graphique sera plus utile après jointures mais on en illustre le fonctionnement :
```python
import numpy as np

plt.figure(figsize=(3,3))
corr = df[["target"]].corr()
sns.heatmap(corr, annot=True, cmap="Blues")
plt.title("Corrélation de la target (dataset initial)")
plt.show()
```
## Interprétation :

La corrélation n’a de sens qu’avec plus de colonnes.

Dans la version finale du dataset (après ajouts de features), cette heatmap permettra :

d’identifier les variables explicatives pertinentes

de détecter la multicolinéarité,

d’orienter le feature engineering.

# 📌Graphique 4 — Distribution cumulée (CDF)

```python
import numpy as np

plt.figure(figsize=(7,4))
sorted_target = np.sort(df["target"])
yvals = np.arange(len(sorted_target)) / float(len(sorted_target)-1)
plt.plot(sorted_target, yvals)
plt.title("Fonction de distribution cumulée – target")
plt.xlabel("target")
plt.ylabel("Probabilité cumulée")
plt.grid()
plt.show()
```
## Interprétation :

La CDF montre une progression régulière, confirmant que le score est étalé dans tout l’intervalle [0,1].

Cela signifie qu’un modèle pourra apprendre des différences fines entre individus.

## 3.3. Modélisation (Machine Learning)

🔧 Modèles testés (3 minimum)

Régression Linéaire

Random Forest Regressor

Gradient Boosting / XGBoost / LightGBM

# 🔁 Validation

Cross-Validation K-Fold (k=5 ou 10)

GridSearchCV / RandomizedSearchCV

# 📊 Exemple de code de modélisation

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

results
```
## 4. Résultats & Discussion

⚠️ À compléter avec tes résultats réels une fois l'entraînement effectué.

 | Modèle              | RMSE | MAE  | R²   |

 | ------------------- | ---- | ---- | ---- |

 | Régression Linéaire | TODO | TODO | TODO |

 | Random Forest       | TODO | TODO | TODO |

 | Gradient Boosting   | TODO | TODO | TODO |

# 4.2. Analyse des résidus (Graphique)
```python

import matplotlib.pyplot as plt

model = GradientBoostingRegressor().fit(X_train, y_train)
preds = model.predict(X_test)
residuals = y_test - preds

plt.figure(figsize=(7,4))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Distribution des résidus")
plt.xlabel("Erreur (y_true - y_pred)")
plt.show()
```
## Interprétation :

Un résidu centré autour de 0 → modèle non biaisé

Dispersion faible → modèle précis

Distribution asymétrique → signe d'underfitting ou d’overfitting selon la forme

## 5. Conclusion

Le dataset DATE.SET.csv constitue une base solide pour un projet complet de régression :

## 🔹 Points forts

Target bien distribuée

Dataset propre

Compatible avec enrichissement (clé client)

Idéal pour ML tabulaire

## 🔹 Limites

Seulement 2 colonnes → nécessite un enrichissement par jointures

Pas d’information métier sur la signification exacte de target

## 🔹 Améliorations possibles

Ajouter des variables comportementales / socio-démographiques

Tester XGBoost et LightGBM

Ajouter SHAP / LIME pour l’explicabilité

Packager le modèle dans une API + pipeline MLOps









