## Compte Rendu – Projet de Modélisation

## Par : EL YESRI Inass

## 1️⃣ Introduction

## 🔹 Contexte

Ce projet a pour objectif de construire un modèle de prédiction à partir d’un dataset réel. Les données initiales sont brutes : elles peuvent contenir des valeurs manquantes, des incohérences, des variables catégorielles, des outliers, etc.

L’enjeu est de transformer ces données en une base exploitable pour entraîner un modèle de Machine Learning fiable.

## 🔹 Problématique

## La problématique principale est la suivante :

Comment obtenir un modèle performant et interprétable à partir de données imparfaites (bruit, valeurs manquantes, potentiellement déséquilibrées) ?

## 🔹 Objectifs

Nettoyer et préparer le dataset.

Choisir et entraîner un ou plusieurs algorithmes de Machine Learning.

Évaluer les modèles via : Accuracy, F1-Score, RMSE, ROC-AUC.

Analyser les erreurs avec la matrice de confusion.

Discuter les limites et proposer des pistes d’amélioration.

## 2️⃣ Méthodologie

## 2.1. Chargement & exploration des données

## import pandas as pd

## # Charger le dataset

df = pd.read_csv("data.csv")  # 👉 à adapter avec le vrai chemin

## # Aperçu des premières lignes

## print(df.head())

## # Infos générales

## print(df.info())

## # Statistiques descriptives

## print(df.describe())

## 🔍 Lecture

head() permet de voir les premières lignes et comprendre la structure.

info() montre les types de variables et les valeurs manquantes.

describe() donne des stats de base (moyenne, min, max, etc.) utiles pour repérer les anomalies.

## 2.2. Gestion des valeurs manquantes

Exemple : imputation simple (numériques → moyenne, catégorielles → mode).

## from sklearn.impute import SimpleImputer

## import numpy as np

## # Séparer features et target

X = df.drop(columns=["target"])  # 👉 adapter le nom de la cible

## y = df["target"]

## # Séparer variables numériques / catégorielles

num_cols = X.select_dtypes(include=np.number).columns

cat_cols = X.select_dtypes(exclude=np.number).columns

## num_imputer = SimpleImputer(strategy="mean")

cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])

## if len(cat_cols) > 0:

    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

## 🧾 Lecture

On évite de supprimer trop de lignes : on préfère imputer.

Moyenne pour les numériques : évite de créer des valeurs extrêmes.

Mode pour les catégorielles : permet de garder une modalité existante.

## 2.3. Encodage & Normalisation

On encode les variables catégorielles et on met les variables à la même échelle (utile pour SVM, régression logistique…).

from sklearn.preprocessing import OneHotEncoder, StandardScaler

## from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

## from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

## # Encodage + scaling dans un pipeline

## preprocessor = ColumnTransformer(

## transformers=[

## ("num", StandardScaler(), num_cols),

        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)

## ]

## )

# Exemple avec un RandomForest (tu peux changer de modèle)

## model = RandomForestClassifier(

## n_estimators=200,

## random_state=42

## )

## pipeline = Pipeline(

## steps=[

## ("preprocess", preprocessor),

## ("model", model)

## ]

## )

## # Split Train/Test

X_train, X_test, y_train, y_test = train_test_split(

## X, y, test_size=0.2, random_state=42, stratify=y

## )

## 🧾 Lecture

Le ColumnTransformer applique différents traitements selon le type de variable.

Le Pipeline garantit que le même prétraitement est appliqué à train et test, ce qui évite les fuites de données.

stratify=y garde le même équilibre entre classes dans train et test.

## 2.4. Entraînement du modèle

## # Entraînement

## pipeline.fit(X_train, y_train)

## 3️⃣ Résultats & Discussion

3.1. Calcul des métriques (Accuracy, F1, ROC-AUC, RMSE)

## from sklearn.metrics import (

## accuracy_score, f1_score,

## roc_auc_score, mean_squared_error

## )

## import numpy as np

## # Prédictions classes

## y_pred = pipeline.predict(X_test)

# Si le modèle supporte predict_proba (RF, LR, etc.)

if hasattr(pipeline.named_steps["model"], "predict_proba"):

    y_proba = pipeline.predict_proba(X_test)[:, 1]  # binaire

## else:

## y_proba = None

## accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, average="weighted")  # "binary" ou "macro" selon le cas

# RMSE sur les classes prédictes (moins courant, mais possible)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ROC-AUC (binaire). Si multi-classe → One-vs-Rest (à adapter).

## if y_proba is not None and len(y.unique()) == 2:

## roc_auc = roc_auc_score(y_test, y_proba)

## else:

## roc_auc = None

## print(f"Accuracy  : {accuracy:.4f}")

## print(f"F1-score  : {f1:.4f}")

## print(f"RMSE      : {rmse:.4f}")

## if roc_auc is not None:

## print(f"ROC-AUC   : {roc_auc:.4f}")

## 🧾 Lecture des métriques

Accuracy : proportion globale de bonnes prédictions.

F1-score : équilibre entre précision et rappel, utile si les classes sont déséquilibrées.

RMSE : racine de l’erreur quadratique moyenne → plus il est faible, mieux c’est (souvent utilisé pour la régression, ici appliqué sur les classes).

ROC-AUC : capacité du modèle à séparer les classes positives et négatives (0.5 = hasard, 1.0 = parfait).

## 3.2. Matrice de confusion + graphique

## from sklearn.metrics import confusion_matrix

## import matplotlib.pyplot as plt

## import seaborn as sns

## cm = confusion_matrix(y_test, y_pred)

## plt.figure(figsize=(6, 5))

## sns.heatmap(

## cm,

## annot=True,

## fmt="d",

## cmap="Blues",

## xticklabels=sorted(y.unique()),

## yticklabels=sorted(y.unique())

## )

## plt.xlabel("Prédictions")

## plt.ylabel("Vérités terrain")

## plt.title("Matrice de confusion")

## plt.tight_layout()

## plt.show()

## 📊 Lecture du graphique – Matrice de confusion

La diagonale représente les bonnes prédictions.

Les valeurs hors diagonale sont les erreurs de classification.

Si une classe est souvent prédite comme une autre, cela révèle :

soit un problème de données (variables pas assez discriminantes),

soit un déséquilibre → le modèle “écrase” les classes minoritaires.

## 3.3. Courbe ROC (si problème binaire)

from sklearn.metrics import roc_curve, roc_auc_score

## if y_proba is not None and len(y.unique()) == 2:

## fpr, tpr, thresholds = roc_curve(y_test, y_proba)

## roc_auc = roc_auc_score(y_test, y_proba)

## plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")  # ligne hasard

## plt.xlabel("Taux de faux positifs (FPR)")

## plt.ylabel("Taux de vrais positifs (TPR)")

## plt.title("Courbe ROC")

## plt.legend(loc="lower right")

## plt.tight_layout()

## plt.show()

## 📊 Lecture du graphique – Courbe ROC

Plus la courbe est au-dessus de la diagonale, plus le modèle sépare bien les classes.

Un AUC proche de 1.0 = excellent, proche de 0.5 = modèle inutile (équivalent au hasard).

Permet de comparer plusieurs modèles indépendamment du seuil de décision.

3.4. Importance des variables (Feature Importance – modèle d’arbres)

## import numpy as np

## model = pipeline.named_steps["model"]

# Attention : pour avoir les bons noms de colonnes après OneHotEncoder,

## # il faut les récupérer depuis le preprocessor :

## feature_names = []

# Noms des variables numériques (après scaling, même nom)

## feature_names.extend(num_cols)

# Noms des variables catégorielles après OneHotEncoder

## if len(cat_cols) > 0:

    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]

    ohe_features = list(ohe.get_feature_names_out(cat_cols))

## feature_names.extend(ohe_features)

## importances = model.feature_importances_

## indices = np.argsort(importances)[::-1]

## top_k = 10  # afficher les 10 plus importantes

top_features = [feature_names[i] for i in indices[:top_k]]

## top_importances = importances[indices[:top_k]]

## plt.figure(figsize=(8, 5))

plt.barh(top_features[::-1], top_importances[::-1])

## plt.xlabel("Importance")

plt.title("Top 10 des features les plus importantes")

## plt.tight_layout()

## plt.show()

📊 Lecture du graphique – Importance des variables

Permet d’identifier les variables qui influencent le plus les prédictions.

Si une variable jugée importante par le domaine métier apparaît très faible ici, cela peut signaler :

## un défaut de preprocessing,

une mauvaise qualité de la donnée pour cette feature,

ou la nécessité de revoir le modèle.

## 4️⃣ Conclusion

## 🔹 Limites du modèle

Dépendance forte à la qualité des données (bruit, valeurs manquantes, erreurs de saisie).

Éventuel déséquilibre entre classes qui pénalise le F1-score et la détection de classes minoritaires.

Modèle possiblement sensible aux hyperparamètres (notamment pour SVM, XGBoost, etc.).

RMSE reste une métrique moins intuitive pour la classification.

## 🔹 Pistes d'amélioration

Améliorer le nettoyage et l'enrichissement des données.

Utiliser une recherche systématique d’hyperparamètres (GridSearchCV, RandomizedSearchCV).

Tester des modèles plus avancés : XGBoost, LightGBM, CatBoost.

Gérer explicitement le déséquilibre des classes (SMOTE, class_weight).

Ajouter des méthodes d’explainability (SHAP, LIME) pour analyser finement les décisions du modèle.