# Guide de Démarrage Rapide - Discrétisation Spatiale

## 📋 Résumé du Projet

**Objectif:** Discrétiser spatialement l'équation de transport 2D au voisinage d'un point de stagnation en préparation pour l'intégration temporelle.

### Équation à résoudre

```
ρ ∂φ/∂t + ρ U·∇φ = Γ ∇²φ
```

- **Champ de vitesse:** u = x, v = -y (écoulement au point de stagnation)
- **Densité:** ρ = 1.2 kg/m³
- **Diffusion:** Γ = 0.1 m²/s
- **Domaine:** [0, 1] × [0, 1]
- **Résolution:** 41 × 41 points

---

## 🚀 Utilisation Rapide

### Installation

```bash
# Les bibliothèques nécessaires sont déjà installées:
# - numpy
# - scipy
# - matplotlib
```

### Exemple Minimal

```python
from transport_differences_finies import TransportDifferencesFinies2D

# 1. Créer le problème
problem = TransportDifferencesFinies2D(
    Lx=1.0, Ly=1.0, nx=41, ny=41,
    rho=1.2, Gamma=0.1
)

# 2. Conditions aux limites
problem.set_boundary_conditions(west=1.0, east=0.0, south=0.0, north=0.0)

# 3. Construire l'opérateur spatial
L, b = problem.build_spatial_operator(scheme='upwind')

# 4. Calculer le flux (après résolution)
flux_dist, flux_total, flux_avg = problem.compute_diffusive_flux_west()
print(f"Flux total: {flux_total}")
```

---

## 📁 Fichiers Fournis

### Code Source

| Fichier | Description |
|---------|-------------|
| `transport_differences_finies.py` | **Code principal** - Classe complète avec toutes les méthodes |
| `exemple_utilisation.py` | **Exemple commenté** - Utilisation pas à pas |
| `analyse_schemas.py` | **Analyse comparative** - Comparaison upwind vs centré |

### Documentation

| Fichier | Description |
|---------|-------------|
| `README.md` | Documentation technique complète |
| `discretisation_spatiale_resume.pdf` | Résumé PDF avec équations et méthodologie |
| `GUIDE_DEMARRAGE_RAPIDE.md` | Ce fichier |

### Visualisations

| Fichier | Contenu |
|---------|---------|
| `matrix_structure.png` | Structure sparse de la matrice L |
| `solution_stationnaire.png` | Solution φ, champ de vitesse, profils |
| `flux_distribution.png` | Distribution du flux à la paroi ouest |

---

## 🔧 Fonctionnalités Clés

### Classe `TransportDifferencesFinies2D`

#### Création et Configuration

```python
problem = TransportDifferencesFinies2D(
    Lx=1.0,      # Longueur en x
    Ly=1.0,      # Longueur en y
    nx=41,       # Points en x
    ny=41,       # Points en y
    rho=1.2,     # Densité
    Gamma=0.1    # Diffusion
)
```

#### Méthodes Principales

```python
# Construction de l'opérateur convection
C = problem.build_convection_operator(scheme='upwind')  # ou 'central'

# Construction de l'opérateur diffusion
D = problem.build_diffusion_operator()

# Opérateur spatial complet
L, b = problem.build_spatial_operator(scheme='upwind')

# Conditions aux limites
problem.set_boundary_conditions(west=1.0, east=0.0, south=0.0, north=0.0)

# Calcul du flux diffusif à la paroi ouest
flux_dist, flux_total, flux_avg = problem.compute_diffusive_flux_west()

# Visualisations
problem.plot_solution(title="Ma Solution")
problem.plot_flux_distribution(flux_dist)
```

---

## 📊 Résultats de Test

### Solution Stationnaire (Vérification)

Résolution de `L·φ = -b` avec:
- φ(x=0, y) = 1.0 (paroi ouest)
- φ = 0 ailleurs sur les frontières

**Résultats:**
- ✓ Flux total: **0.590851**
- ✓ Flux moyen: **0.590851**
- ✓ Flux max: **3.018101** à y=0.975
- ✓ φ min = 0, φ max = 1

### Structure de la Matrice L

- Taille: **1521 × 1521**
- Éléments non-nuls: **7449**
- Densité: **0.32%**
- Type: Pentadiagonale par blocs (sparse)

---

## 🔢 Nombres Adimensionnels

### Nombre de Péclet

```
Pe = ρ u Δx / Γ
```

- Pe_x = 0.3
- Pe_y = 0.3
- **Pe < 2** → Les deux schémas sont stables ✓

### Critère CFL (pour intégration temporelle)

Pour Euler explicite:
- Convection: Δt < 0.0125 s
- Diffusion: Δt < 0.001875 s
- **Recommandé: Δt < 0.001875 s**

---

## 📐 Schémas de Discrétisation

### Convection (Schéma Upwind)

**Avantages:**
- ✓ Inconditionnellement stable
- ✓ Pas d'oscillations
- ✓ Robuste

**Inconvénients:**
- Ordre 1 (moins précis)
- Diffusion numérique

### Diffusion (Schéma Centré)

**Différences finies d'ordre 2:**

```
∂²φ/∂x² ≈ (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1}) / Δx²
∂²φ/∂y² ≈ (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j}) / Δy²
```

---

## 🎯 Prochaines Étapes

### Pour l'Intégration Temporelle

1. **Choisir une méthode:**
   - Euler explicite (simple)
   - Runge-Kutta 4 (recommandé)
   - Méthode implicite (très stable)

2. **Implémenter la boucle:**

```python
from scipy.integrate import solve_ivp

def rhs(t, phi_interior):
    return -1.0 / problem.rho * (L @ phi_interior + b)

# Condition initiale
phi0 = problem.phi_to_interior()  # φ = 0 partout

# Intégration
sol = solve_ivp(
    rhs, 
    t_span=(0, 0.12),  # De t=0 à t=0.12s
    y0=phi0,
    method='RK45',
    max_step=0.001     # Respecter le critère CFL
)

# Solution finale
phi_final = sol.y[:, -1]
problem.interior_to_phi(phi_final)

# Flux au temps final
flux_dist, flux_total, flux_avg = problem.compute_diffusive_flux_west()
print(f"Flux à t=0.12s: {flux_total}")
```

3. **Analyser:**
   - Évolution de φ(x,y,t)
   - Évolution du flux en fonction du temps
   - Convergence (si applicable)

---

## 💡 Conseils d'Utilisation

### Performance

- La matrice L est **sparse** → utilisez `scipy.sparse`
- Pour des grilles plus fines, augmentez nx et ny
- Le schéma upwind est plus stable mais moins précis

### Débogage

```python
# Vérifier la matrice
print(f"Matrice L: {L.shape}")
print(f"Non-zéros: {L.nnz}")

# Vérifier le vecteur source
print(f"Vecteur b: {b.shape}")
print(f"Norme de b: {np.linalg.norm(b)}")

# Afficher un résumé
problem.print_summary()
```

### Visualisation

```python
# Solution complète
problem.plot_solution(
    title="Ma solution",
    save_path="ma_solution.png"
)

# Flux uniquement
problem.plot_flux_distribution(
    flux_dist,
    save_path="mon_flux.png"
)
```

---

## ❓ FAQ

**Q: Pourquoi le schéma upwind?**
A: Il est inconditionnellement stable pour la convection, ce qui évite les oscillations non physiques.

**Q: Peut-on utiliser le schéma centré?**
A: Oui, si Pe < 2 (notre cas). Il sera plus précis mais peut osciller pour Pe > 2.

**Q: Comment changer la résolution?**
A: Modifiez nx et ny lors de la création: `TransportDifferencesFinies2D(nx=61, ny=61, ...)`

**Q: Comment changer les conditions aux limites?**
A: Utilisez `problem.set_boundary_conditions(west=..., east=..., south=..., north=...)`

**Q: Le code fonctionne-t-il pour d'autres champs de vitesse?**
A: Non, il est spécifique à u=x, v=-y. Pour d'autres champs, modifiez les méthodes de calcul de u et v.

---

## 📞 Support

Pour toute question sur le code:
1. Consultez le `README.md` pour les détails techniques
2. Regardez `exemple_utilisation.py` pour un exemple complet
3. Lisez le PDF pour la méthodologie mathématique

---

## ✅ Checklist de Vérification

Avant l'intégration temporelle, vérifiez:

- [ ] La matrice L est bien construite (1521×1521)
- [ ] Le vecteur b a la bonne taille (1521)
- [ ] Les conditions aux limites sont correctement appliquées
- [ ] Le flux stationnaire semble raisonnable
- [ ] Le critère CFL est respecté (Δt < 0.001875 s)

---

**Date de création:** 11 février 2026  
**Version:** 1.0  
**Statut:** ✅ Prêt pour l'intégration temporelle
