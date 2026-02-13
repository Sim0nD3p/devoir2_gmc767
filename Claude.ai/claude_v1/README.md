# Discrétisation Spatiale par Différences Finies - Équation de Transport 2D

## Description du Problème

Ce code implémente la discrétisation spatiale de l'équation de transport 2D au voisinage d'un point de stagnation :

**Équation:** ρ ∂φ/∂t + ρ U·∇φ = Γ ∇²φ

**Champ de vitesse:** u = x, v = -y (écoulement au point de stagnation)

**Paramètres:**
- ρ (rho) = 1.2 kg/m³ (densité du fluide)
- Γ (Gamma) = 0.1 m²/s (coefficient de diffusion)

## Fichiers Fournis

### 1. `transport_differences_finies.py` (Programme Principal)

Code complet avec la classe `TransportDifferencesFinies2D` qui implémente:

#### Fonctionnalités principales:

**a) Discrétisation de la convection:**
- `build_convection_operator(scheme='upwind')`: Construit l'opérateur ρ U·∇φ
- Schémas disponibles:
  - `'upwind'`: Décentré amont (stable, diffusif numériquement)
  - `'central'`: Centré (ordre 2, peut osciller si Péclet > 2)

**b) Discrétisation de la diffusion:**
- `build_diffusion_operator()`: Construit l'opérateur Γ ∇²φ
- Utilise des différences finies centrées d'ordre 2:
  - ∂²φ/∂x² ≈ (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1}) / Δx²
  - ∂²φ/∂y² ≈ (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j}) / Δy²

**c) Opérateur spatial complet:**
- `build_spatial_operator(scheme)`: Combine convection et diffusion
- Retourne la matrice L et le vecteur b pour: L·φ = -b (problème stationnaire)
- Pour l'intégration temporelle: dφ/dt = -1/ρ * (L·φ + b)

**d) Calcul du flux diffusif:**
- `compute_diffusive_flux_west()`: Calcule le flux à la paroi ouest
- Flux = -Γ ∂φ/∂x|_{x=0}
- Utilise une différence finie décentrée avant d'ordre 2

### 2. `analyse_schemas.py` (Analyse Comparative)

Script qui compare différents schémas de discrétisation et analyse:
- Comparaison upwind vs centré
- Analyse du nombre de Péclet
- Étude de convergence avec raffinement de grille
- Visualisations détaillées

## Utilisation du Code

### Exemple de base:

```python
from transport_differences_finies import TransportDifferencesFinies2D

# Création du problème
problem = TransportDifferencesFinies2D(
    Lx=1.0,      # Longueur du domaine en x
    Ly=1.0,      # Longueur du domaine en y
    nx=41,       # Nombre de points en x
    ny=41,       # Nombre de points en y
    rho=1.2,     # Densité
    Gamma=0.1    # Coefficient de diffusion
)

# Définir les conditions aux limites
problem.set_boundary_conditions(
    west=1.0,   # φ = 1 à x=0
    east=0.0,   # φ = 0 à x=Lx
    south=0.0,  # φ = 0 à y=0
    north=0.0   # φ = 0 à y=Ly
)

# Construire l'opérateur spatial
L, b = problem.build_spatial_operator(scheme='upwind')

# Pour résolution stationnaire:
from scipy.sparse.linalg import spsolve
phi_interior = spsolve(L, -b)
problem.interior_to_phi(phi_interior)

# Calculer le flux à la paroi ouest
flux_dist, flux_total, flux_avg = problem.compute_diffusive_flux_west()
print(f"Flux total: {flux_total:.6f}")
print(f"Flux moyen: {flux_avg:.6f}")
```

### Pour l'intégration temporelle (prochaine étape):

```python
# L'opérateur spatial L et le vecteur b sont prêts
# Équation différentielle à résoudre:
# dφ/dt = RHS(φ) = -1/ρ * (L·φ + b)

def rhs_function(t, phi_interior):
    """Membre de droite pour l'intégration temporelle"""
    return -1.0 / problem.rho * (L @ phi_interior + b)

# Utiliser scipy.integrate.solve_ivp ou une méthode explicite
# pour intégrer de t=0 à t=0.12 s
```

## Structure de la Discrétisation

### Grille:
- Domaine: [0, Lx] × [0, Ly]
- Grille uniforme avec nx × ny points
- Points intérieurs: (nx-2) × (ny-2)
- Conditions de Dirichlet aux frontières

### Numérotation des points:
- Convention: i = indice en y (lignes), j = indice en x (colonnes)
- Indice 1D: idx = (i-1)*(nx-2) + (j-1) pour les points intérieurs
- Frontières stockées séparément dans problem.phi[i, j]

### Structure de la matrice L:
- Matrice sparse (pentadiagonale par blocs)
- Taille: (nx-2)*(ny-2) × (nx-2)*(ny-2)
- Chaque ligne a au maximum 5 éléments non nuls
- Densité typique: ~0.3% pour une grille 41×41

## Résultats de Test

Pour le problème de test avec:
- Domaine: [0, 1] × [0, 1]
- Grille: 41 × 41
- Conditions: φ(0, y) = 1, φ ailleurs aux frontières = 0

**Résultats (solution stationnaire):**
- Flux total à la paroi ouest: 0.590851
- Flux moyen par unité de longueur: 0.590851
- Flux maximum: 3.018101 à y = 0.975

## Nombres Adimensionnels

### Nombre de Péclet:
Pe = ρ * u * Δx / Γ

Pour u_max ≈ 1.0 et Δx = 0.025:
- Pe_x ≈ 1.2 × 1.0 × 0.025 / 0.1 = 0.3

**Critère:** Pe < 2 pour stabilité du schéma centré
→ Les deux schémas (upwind et centré) sont stables

### Nombre de Courant (pour intégration temporelle):
CFL = u * Δt / Δx

À déterminer lors du choix du pas de temps Δt.

## Visualisations Générées

1. **matrix_structure.png**: Structure sparse de la matrice L
2. **solution_stationnaire.png**: Solution φ, champ de vitesse, et profils
3. **flux_distribution.png**: Distribution du flux le long de la paroi ouest

## Prochaines Étapes (Intégration Temporelle)

Le code actuel fournit la discrétisation spatiale complète. Pour résoudre le problème instationnaire jusqu'à t = 0.12 s, il faudra:

1. Choisir une méthode d'intégration temporelle:
   - Euler explicite (simple mais restrictif en Δt)
   - Runge-Kutta 4 (RK4) - bon compromis
   - Méthodes implicites (inconditionnellement stable)

2. Implémenter la boucle temporelle:
   ```python
   t = 0
   t_final = 0.12
   dt = ... # à déterminer selon critère de stabilité
   
   while t < t_final:
       # Intégration d'un pas de temps
       phi_interior_new = time_step(phi_interior_old, dt, L, b)
       # Mise à jour
       phi_interior_old = phi_interior_new
       t += dt
   ```

3. Calculer le flux final à t = 0.12 s

## Notes Importantes

- **Stabilité:** Le schéma upwind est inconditionnellement stable pour la convection
- **Précision:** Le schéma upwind est d'ordre 1, le centré est d'ordre 2
- **Diffusion numérique:** Le schéma upwind introduit de la diffusion numérique
- **Choix recommandé:** Upwind pour Péclet > 2, centré sinon

## Auteur et Licence

Code développé pour la résolution de l'équation de transport 2D.
Usage académique et éducatif.
