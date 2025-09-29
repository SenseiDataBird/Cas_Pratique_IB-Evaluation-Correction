# ============================================================================
# ------------ IMPORTATION DES PACKAGES ET DE LA DONNÉE ----------------------
# ============================================================================

import pandas as pd # Pour la lecture des données
import numpy as np # Pour les calculs mathématiques

import matplotlib.pyplot as plt # Pour les graphiques

from sklearn.model_selection import train_test_split, cross_val_score, KFold # Pour la division des données et validation croisée
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Pour les métriques d'évaluation

import xgboost as xgb # Pour la construction du modèle

# ---------------------------------------------------------------------------

# Chargement des données prétraitées
df = pd.read_csv('data/house_pred_for_ml.csv')

# Séparation des features (X) et de la target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Division en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================================
# -------------------- CONSTRUCTION DU MODÈLE --------------------------------
# ============================================================================

# Création et entraînement du modèle XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,        # Nombre d'arbres dans l'ensemble
    learning_rate=0.1,       # Taux d'apprentissage
    max_depth=5,             # Profondeur maximale de chaque arbre
    min_child_weight=1,      # Poids minimum requis dans une feuille
    subsample=0.8,           # Fraction d'échantillons utilisés par arbre
    colsample_bytree=0.8,    # Fraction de features utilisées par arbre
    random_state=42          # Graine pour la reproductibilité
)

# ---------------------------------------------------------------------------

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Le modèle est maintenant prêt pour l'évaluation
print("Modèle XGBoost entraîné avec succès !")

# ============================================================================
# -------------------- ÉVALUATION DU MODÈLE -------------------------------
# ============================================================================

# Prédictions sur train et test
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# Calcul des métriques de base
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
mae_test = mean_absolute_error(y_test, y_pred)

print("="*60)
print("RAPPORT D'ÉVALUATION DU MODÈLE XGBOOST")
print("="*60)

# ============================================================================
# ----------------- DÉTECTION DU SURAPPRENTISSAGE ----------------------------
# ============================================================================

print(f"\n1. Détection du surapprentissage (fiabilité du modèle) :")
print("-" * 50)
print(f"R² Train: {r2_train:.3f} | R² Test: {r2_test:.3f} | Écart: {r2_train - r2_test:.3f}")
print(f"RMSE Train: {rmse_train:.0f} | RMSE Test: {rmse_test:.0f}") 

if r2_train - r2_test > 0.1:
    print("Surapprentissage détecté - Modèle non fiable")
else:
    print("Pas de surapprentissage - Modèle fiable")

# ============================================================================
# -------------------- VALIDATION CROISÉE ------------------------------------
# ============================================================================

print(f"\n2. Validation croisée (robustesse du modèle) :")
print("-" * 50)

r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"R² moyen : {r2_scores.mean():.3f} (+/- {r2_scores.std() * 2:.3f})")
print(f"Coefficient de variation : {(r2_scores.std() / r2_scores.mean()):.1%}")

if (r2_scores.std() / r2_scores.mean()) < 0.05:
    print("Modèle stable - Performances consistantes")
else:
    print("Modèle instable - Performances variables")

# ============================================================================
# -------------------- MÉTRIQUES DE PERFORMANCE ------------------------------
# ============================================================================

print(f"\n3. Métriques de performance (qualité des prédictions) :")
print("-" * 50)
print(f"R² (determination coefficient) : {r2_test:.3f}")
print(f"   → Le modèle explique {r2_test:.1%} de la variance des prix")
print(f"RMSE (mean squared error) : {rmse_test:.0f} euros")
print(f"   → Erreur moyenne de {rmse_test:.0f} euros")
print(f"MAE (mean absolute error) : {mae_test:.0f} euros")
print(f"   → Erreur moyenne de {mae_test:.0f} euros")

# ============================================================================
# ------------------------- ANALYSE DES FEATURES -----------------------------
# ============================================================================

print(f"\n4. Importance des features (interprétabilité) :")
print("-" * 50)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("Top 5 des features les plus importantes :")
print(feature_importance.head().to_string(index=False))

# ============================================================================
# -------------------- VISUALISATIONS DIAGNOSTIQUES -------------------------
# ============================================================================

print(f"\n5. Diagnostiques visuelles :")
print("-" * 50)

# Graphique de dispersion (le plus important)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Prix réel')
plt.ylabel('Prix prédit')
plt.title('Prix réel vs Prix prédit')

# Graphique des résidus
plt.subplot(1, 2, 2)
plt.scatter(y_pred, y_test - y_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prix prédit')
plt.ylabel('Résidus')
plt.title('Analyse des résidus')
plt.tight_layout()
plt.show()

# Graphique de validation croisée
plt.figure(figsize=(8, 6))
plis = range(1, 6)
plt.plot(plis, r2_scores, 'o-', linewidth=2, markersize=8, color='blue')
plt.axhline(y=r2_scores.mean(), color='red', linestyle='--', label=f'Moyenne: {r2_scores.mean():.3f}')
plt.xlabel('Numéro du pli')
plt.ylabel('Score R²')
plt.title('Stabilité du modèle - Validation Croisée')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(plis)
plt.show()

# ============================================================================
# -------------------- CONCLUSION GÉNÉRALE ----------------------------------
# ============================================================================

print(f"\n Conclusion générale :")
print("="*60)

# Score global
if r2_test > 0.8 and (r2_train - r2_test) < 0.1 and (r2_scores.std() / r2_scores.mean()) < 0.05:
    print("Modèle excellent - Prêt pour la production")
elif r2_test > 0.7 and (r2_train - r2_test) < 0.15:
    print("Modèle correct - Améliorations possibles")
else:
    print("Modèle à retravailler - Problèmes détectés")

print(f"\n Résumé des performances :")
print(f"• Précision : {r2_test:.1%} de variance expliquée")
print(f"• Erreur : ±{rmse_test:.0f} euros en moyenne")
print(f"• Fiabilité : {'Stable' if (r2_scores.std() / r2_scores.mean()) < 0.05 else 'Instable'}")
print(f"• Surapprentissage : {'Non' if (r2_train - r2_test) < 0.1 else 'Oui'}")