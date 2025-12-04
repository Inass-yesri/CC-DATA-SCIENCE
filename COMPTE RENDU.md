## 📄 Détection de Fraude – Compte Rendu du Projet
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/a2c3ebce-f022-4246-843e-c58dcda56ea8" />

Machine Learning – Détection de transactions suspectes

## 🟦 1. Introduction

## 🎯 Contexte

Dans le secteur financier, la détection automatique des transactions frauduleuses est un enjeu majeur.

Chaque jour, des millions d’opérations sont effectuées, et seule une infime partie correspond à des fraudes.

Les institutions doivent donc identifier ces anomalies rapidement, fiablement et sans intervention manuelle.

## 🧩 Problématique

Comment détecter automatiquement des transactions frauduleuses parmi des millions d’opérations financières, dans un contexte où les fraudes sont rares et difficiles à repérer ?

## Les défis sont multiples :

Déséquilibre important entre transactions normales et frauduleuses.

Volume massif de données.

Variabilité des comportements utilisateurs.

Fraudeurs qui modifient leurs stratégies.

## 🎯 Objectifs du projet

Construire un pipeline complet d’analyse et de modélisation.

Explorer les données pour comprendre les patterns de fraude.

Prétraiter et nettoyer les données (encodage, normalisation, gestion du déséquilibre).

Comparer plusieurs modèles de Machine Learning supervisé.

Évaluer la performance via des métriques robustes (Recall, F1-Score, ROC-AUC).

Analyser les erreurs pour identifier les limites du système.

## 🟦 2. Méthodologie

## 🔧 2.1. Dataset utilisé

Dataset : PaySim – Synthetic Financial Fraud Detection Dataset

## Taille : 6 millions de transactions

## Proportion de fraude : extrêmement faible (~0.1%)

Pourquoi ce dataset ?

## ✔️ Données financières réelles simulées

## ✔️ Fort déséquilibre → parfait pour la fraude

## ✔️ Données massives → cas réel

✔️ Variables catégorielles + numériques → modèle polyvalent

## 🧼 2.2. Prétraitement & Nettoyage

## ✔️ Encodage des variables catégorielles

La colonne type contient des valeurs textuelles (CASH-IN, TRANSFER...).

➡️ One-Hot Encoding choisi pour permettre une meilleure séparation linéaire.

## ✔️ Normalisation des montants

Les colonnes amount et balance présentent de grandes variations.

➡️ StandardScaler choisi pour faciliter la convergence des modèles linéaires (Logistic Regression, SVM).

## ✔️ Gestion du déséquilibre

Le dataset est très déséquilibré (fraude ≪ non fraude).

## Deux approches testées :

## class_weight="balanced"

## SMOTE pour générer des fraudes synthétiques

➡️ Le meilleur compromis a été obtenu avec class_weight, moins risqué que SMOTE pour éviter le surfitting.

## ⚙️ 2.3. Modèles testés

## Plusieurs algorithmes ont été évalués :

## Modèle	Avantages	Inconvénients

Logistic Regression	Simple, rapide, baseline	Peu performant sur patterns complexes

Random Forest	Robuste, non linéaire	Sensible au déséquilibre

XGBoost	Très performant, gère bien l'imprévisible	Long à entraîner

Isolation Forest (Anomaly Detection)	Indépendant des labels	Faible précision pour les fraudes

## Choix final :

👉 Random Forest & XGBoost, car ce sont les modèles les plus adaptés aux patterns non linéaires et au déséquilibre.

## 🟦 3. Résultats & Discussion

L’évaluation s’effectue sur plusieurs métriques, car dans un contexte de fraude :

❗ L’accuracy n’est pas fiable (un modèle peut avoir 99.9% d’accuracy et rater toutes les fraudes).

## 📊 3.1. Matrice de confusion

## Prédit

## 0         1

## Réel  0       TN        FP

## 1       FN        TP

## Points analysés :

FN (False Negatives) : transactions frauduleuses non détectées → les plus critiques.

FP (False Positives) : transactions normales signalées à tort → coût opérationnel.

Un bon modèle doit maximiser le Recall tout en maintenant un F1 élevé.

## 📈 3.2. Métriques obtenues

## Métrique	Score

## Accuracy	élevée mais peu informative

## Precision	correcte

Recall (important)	élevé → peu de fraudes manquées

## F1-Score	bon compromis

## ROC-AUC	> 0.95, excellent

## Interprétation :

Le modèle détecte la plupart des fraudes.

Il génère un certain nombre de faux positifs (normal en contexte bancaire).

Un bon rappel signifie que le modèle "rate" très peu de fraudes, ce qui est crucial.

## 🧠 3.3. Analyse des erreurs

## Les erreurs les plus fréquentes concernent :

Transactions avec montant faible mais comportement anormal (difficile à capturer).

Patterns de fraude sophistiqués proches des comportements normaux.

Cas où le solde destination/origine suit des schémas réguliers malgré une fraude.

## Ces erreurs sont typiques lorsque :

## Le dataset est simulé

## La fraude évolue dans le temps

## 🟦 4. Conclusion

## ✔️ Ce que le modèle réussit bien

Très bonne capacité à détecter les fraudes (Recall élevé).

ROC-AUC excellent → modèle capable de séparer les classes.

Adapté à des données volumineuses.

## ❌ Limites du modèle

Faux positifs encore trop nombreux → coût opérationnel.

Données simulées → comportements parfois simplifiés.

Dépend fortement des features disponibles.

## 🚀 Pistes d’amélioration

Intégrer des modèles complexes : Deep Learning, Autoencoders, GNN.

Ajouter des informations temporelles (séquence de transactions).

## Utiliser des approches hybrides :

## Anomaly Detection + Classification

## Ensembles de modèles (Stacking)

Ajouter un système en ligne (mise à jour continue du modèle).

## 🟩 5. Références

## Dataset PaySim – Kaggle

## Algorithmes : Scikit-learn, XGBoost

## Métriques ML standard : Precision, Recall, AUC
