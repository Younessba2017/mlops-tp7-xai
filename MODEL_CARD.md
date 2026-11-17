# Model Card - fraude, version 1

## Modèle
Registre MLflow : fraude, version 1, alias production.
RandomForestClassifier, 500 arbres, profondeur maximale 16, random_state=42.
Run d'origine : 97f22344da7341d9a7c7ce5adc7504a7.

## Objectif et usage prévu
Détecter les transactions potentiellement frauduleuses pour aider un
conseiller à examiner les paiements suspects dans le cadre de ce TP.

## Usages exclus
Bloquer définitivement un compte sans revue humaine.
Utiliser directement ce modèle pédagogique sur des paiements réels
sans validation sur des données représentatives.

## Données
Données synthétiques du TP2, fichier data/transactions.csv.
5 000 transactions, dont 568 fraudes, soit 11,36 %.
Variables : montant, anciennete, freq_24h et nuit. Cible : fraude.
Empreinte MD5 : cf9f6f86c141e7df32c61715c493483f.

## Performances
Jeu de test : 1 250 transactions, dont 134 fraudes.
Découpage : test_size=0.25, random_state=42.
F1 : 0,8669. Précision : 0,8837. Rappel : 0,8507. AUC : 0,9872.
114 fraudes détectées, 20 fraudes manquées et 15 fausses alertes.

## Méthodes d'explicabilité
Globale : permutation sur le test, 10 répétitions, et SHAP sur
400 transactions sélectionnées avec random_state=0.
Locale : SHAP et LIME, avec 2 000 voisins et graine 0 pour la référence.
SHAP classe les variables par importance moyenne absolue dans cet ordre :
montant, anciennete, nuit, freq_24h.

## Explication du paiement de Mme Benali
Le modèle estime une probabilité de fraude de 92,40 %.
Le paiement de nuit (+0,3697) et le montant de 1 800 MAD (+0,3665)
contribuent principalement à ce score.
Valeur de base + contributions SHAP = probabilité prédite.

## Comparaison avec un modèle interprétable
La régression logistique avec log10(montant) obtient un F1 de 0,8593
et détecte 113 fraudes sur 134, avec des coefficients directement lisibles.
L'arbre de profondeur 3 possède 8 feuilles et des règles courtes,
mais son F1 est de 0,5737, avec 72 fraudes détectées.

## Limites et risques connus
Les données synthétiques limitent la généralisation aux paiements réels.
Le modèle manque 20 fraudes et génère 15 fausses alertes sur le test.
LIME varie selon la graine, notamment pour freq_24h.
Les contributions SHAP décrivent le modèle et ne prouvent pas une causalité.

## Recommandations d'utilisation et de surveillance
Prévoir une revue humaine des alertes et suivre précision, rappel et F1.
Surveiller l'évolution des données et recalculer les explications après
chaque réentraînement ou changement de version de production.

## Traçabilité
Expérience MLflow : tp7-explicabilite.
Run des explications : d4045c9e1267498394ebbdfa148bfcf8.
La version 1 porte le tag explications_run vers ce run.
Les graphiques et résultats sont archivés dans le dossier explications.
Auteurs : Youness Bachar et Zahdal El Mahdi.
Date : 17 novembre 2026.
Prochaine revue : après réentraînement ou changement de version.