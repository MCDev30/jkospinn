# JKO-SPINN: Apprentissage de modèles SDE et identification de paramètres physiques avec Score-Based PINNs et JKO

---

## Aperçu

**JKO-SPINN** est une implémentation moderne d’un algorithme innovant pour l’apprentissage de Systèmes Dynamiques Stochastiques (SDEs) depuis des trajectoires bruitées. Combinant la puissance des réseaux de neurones à score (score-based models), l’intégration de connaissances physiques via des Equation Differentiales Stochastiques (EDS) et des techniques avancées d’optimisation inspirées du schéma de Jordan-Kinderlehrer-Otto (JKO), ce code permet l’inférence jointe:

- **du champ de drift inconnu (score)**
- **des paramètres physiques du processus**  
- **à partir de données partiellement observées et bruitées**

L’approche repose sur la résolution d’un problème variationnel motivé mathématiquement, avec, en cœur du modèle, un réseau de neurones physique-informé (PINNs) qui apprend une fonction score ajustant à la fois la data et la physique sous-jacente à l’aide d’opérateurs différentielles (scores, divergences, Hutchinson trace, etc.).

---

## Fonctionnalités principales

- Simulateur generique de SDE (Euler-Maruyama)
- Prise en charge de processus de type Ornstein-Uhlenbeck et double puits ("double-well")
- Génération automatique de trajectoires de références pour l’évaluation
- Réseau de neurones score-based avec Fourrier Features et Swish activation
- Perte hybride combinant:  
  - **Score Matching** (data-driven)  
  - **Physique** (PINN, opérateur différentiel résiduel)  
  - **Guidage L2 sur les vrais paramètres (optionnel)**
- Optimisation jointe des paramètres réseaux & physiques
- Résultats entièrement traçables (losses, courbes convergence, erreurs relatives, etc.)
- Expérimentations pour évaluer la robustesse (sparsité, ablations) et la stabilité (multi-initialisation)
- Visualisations scientifiques: courbes, barplots, tableaux de synthèse…

---

## Guide d'utilisation

### 1. Prérequis

- JAX (`jax`, `jaxlib`)
- NumPy, Matplotlib, Seaborn, tqdm
- optax
- scipy (pour les stats)

Tout le code est compatible GPU si JAX l’est sur votre machine.

### 2. Structure du code

- `Config`: configuration générale des hyperparamètres et expérimentation
- `OUProcess` & `DoubleWellProcess`: définitions des SDEs jouet
- `SDESimulator`: simulation SDE générique (Euler-Maruyama)
- `generate_data`: génération de jeux de données synthétiques bruités
- `score_network`, `init_network`: architecture du score-based PINN avec features Fourier
- Opérateurs physiques: drift, divergences, trace (Hutchinson)
- Fonction de losses: data (DSM), physique (PINN), guidage paramètre
- Boucle d’entraînement `train_jko_spinn`
- Visualisations des résultats et analyses

### 3. Exécution des expériences

Expériences typiques :

- **EXP 1 : Processus d'Ornstein-Uhlenbeck**  
  *Apprentissage des paramètres θ, μ, σ du processus OU à partir de trajectoires bruitées*

- **EXP 2 : Processus Double-Puits**  
  *Inférence du paramètre α et du bruit sur un potentiel double-puits*

- **EXP 3 : Multi-Initialisation & robustesse**  
  *Exécutions multiples pour évaluer la stabilité et la distribution des erreurs*

Chaque expérience génère des synthèses des performances : erreurs relatives de chaque paramètre, convergence, courbes de loss et visualisations.

### 4. Résultats et visualisation

Des fonctions de visualisation dédiées permettent :

- Suivi de la convergence des paramètres physiques
- Visualisation des courbes de pertes (loss curves)
- Tableaux de synthèse et barplots d’erreurs relatives (%)
- Analyses supplémentaires : sparsité des données, ablations du nombre d’échantillons Hutchinson, λ_physics, etc.

---

Les paramètres, la génération des jeux de données et le choix du SDE sont complètement configurables via la classe `Config`.

---

## Points clés algorithmique

- Optimisation conjointe PINN & paramètres SDE physiques par descente de gradient (AdamW/Optax)
- Score-based learning par *denoising score matching* (DSM)
- Imposition de la physique via opérateur différentiel (JKO-PINN residual)
- Utilisation du trace de Hutchinson pour efficacité sur la divergence du score
- Stabilité : early stopping, historique, multi-restarts

---

## Auteurs

Développé par :

**Charbel MAMLANKOU** & **al.**  
charbelzeusmamlankou@gmail.com, École Nationale Supérieure de Génie Mathématique et Modélisation
