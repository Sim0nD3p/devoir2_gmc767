"""
Exemple d'utilisation simple du code de discrétisation spatiale
"""

from transport_differences_finies import TransportDifferencesFinies2D
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION DU PROBLÈME
# ============================================================================

print("Configuration du problème de transport 2D...")
print("="*60)

# Création de l'objet problème
problem = TransportDifferencesFinies2D(
    Lx=1.0,      # Domaine: [0, 1] m
    Ly=1.0,      
    nx=41,       # Résolution: 41×41 points
    ny=41,
    rho=1.2,     # Densité du fluide
    Gamma=0.1    # Coefficient de diffusion
)

print(f"Domaine: [0, {problem.Lx}] × [0, {problem.Ly}]")
print(f"Résolution: {problem.nx} × {problem.ny}")
print(f"Δx = {problem.dx:.5f} m, Δy = {problem.dy:.5f} m")
print(f"ρ = {problem.rho}, Γ = {problem.Gamma}")

# ============================================================================
# CONDITIONS AUX LIMITES
# ============================================================================

print("\n" + "="*60)
print("Conditions aux limites (Dirichlet):")
print("="*60)

problem.set_boundary_conditions(
    west=1.0,    # φ = 1 à x = 0 (paroi chaude)
    east=0.0,    # φ = 0 à x = Lx
    south=0.0,   # φ = 0 à y = 0
    north=0.0    # φ = 0 à y = Ly
)

print("  Ouest (x=0): φ = 1.0")
print("  Est (x=Lx): φ = 0.0")
print("  Sud (y=0): φ = 0.0")
print("  Nord (y=Ly): φ = 0.0")

# ============================================================================
# CONSTRUCTION DE L'OPÉRATEUR SPATIAL
# ============================================================================

print("\n" + "="*60)
print("Construction de l'opérateur spatial...")
print("="*60)

# Choisir le schéma: 'upwind' ou 'central'
scheme = 'upwind'
print(f"Schéma de convection: {scheme.upper()}")

# Construction de l'opérateur L et du vecteur source b
L, b = problem.build_spatial_operator(scheme=scheme)

print(f"Matrice L: {L.shape[0]} × {L.shape[1]}")
print(f"Éléments non-nuls: {L.nnz}")
print(f"Densité: {L.nnz / (L.shape[0]**2) * 100:.2f}%")

# ============================================================================
# RÉSOLUTION DU PROBLÈME STATIONNAIRE (TEST)
# ============================================================================

print("\n" + "="*60)
print("Résolution du problème stationnaire (test)...")
print("="*60)
print("Équation: ρ U·∇φ = Γ ∇²φ")
print("Système linéaire: L·φ = -b")

# Résolution du système linéaire
phi_interior_steady = spsolve(L, -b)

# Mise à jour de la solution
problem.interior_to_phi(phi_interior_steady)

print(f"✓ Solution calculée")
print(f"  φ min = {problem.phi.min():.6f}")
print(f"  φ max = {problem.phi.max():.6f}")
print(f"  φ moyen = {problem.phi.mean():.6f}")

# ============================================================================
# CALCUL DU FLUX DIFFUSIF À LA PAROI OUEST
# ============================================================================

print("\n" + "="*60)
print("Calcul du flux diffusif à la paroi ouest...")
print("="*60)
print("Flux = -Γ ∂φ/∂x|_{x=0}")

flux_dist, flux_total, flux_avg = problem.compute_diffusive_flux_west()

print(f"\n  Flux total: {flux_total:.6f}")
print(f"  Flux moyen: {flux_avg:.6f}")
print(f"  Flux max: {flux_dist.max():.6f} à y={problem.y[np.argmax(flux_dist)]:.3f}")
print(f"  Flux min: {flux_dist.min():.6f} à y={problem.y[np.argmin(flux_dist)]:.3f}")

# ============================================================================
# VISUALISATION
# ============================================================================

print("\n" + "="*60)
print("Génération des visualisations...")
print("="*60)

# 1. Solution et champ de vitesse
fig1 = problem.plot_solution(
    title="Solution stationnaire φ (test)",
    save_path='/mnt/user-data/outputs/exemple_solution.png'
)

# 2. Distribution du flux
fig2 = problem.plot_flux_distribution(
    flux_dist,
    save_path='/mnt/user-data/outputs/exemple_flux.png'
)

print("✓ Figures sauvegardées:")
print("  - exemple_solution.png")
print("  - exemple_flux.png")

# ============================================================================
# PRÉPARATION POUR L'INTÉGRATION TEMPORELLE
# ============================================================================

print("\n" + "="*60)
print("Préparation pour l'intégration temporelle...")
print("="*60)

# Extraction des valeurs intérieures
phi_interior = problem.phi_to_interior()

print(f"Vecteur φ_interior: shape = {phi_interior.shape}")
print(f"\nPour l'intégration temporelle, utiliser:")
print(f"  dφ/dt = -1/ρ * (L·φ + b)")
print(f"  où ρ = {problem.rho}")

# Calcul du RHS à l'instant initial (φ = 0 partout sauf frontières)
problem.phi = np.zeros((problem.ny, problem.nx))
problem.set_boundary_conditions(west=1.0, east=0.0, south=0.0, north=0.0)
phi_interior_init = problem.phi_to_interior()

rhs_init = -1.0 / problem.rho * (L @ phi_interior_init + b)

print(f"\nÀ t=0 (φ=0 partout sauf frontières):")
print(f"  ||RHS|| = {np.linalg.norm(rhs_init):.6e}")
print(f"  RHS max = {rhs_init.max():.6e}")
print(f"  RHS min = {rhs_init.min():.6e}")

# ============================================================================
# CRITÈRES DE STABILITÉ POUR L'INTÉGRATION TEMPORELLE
# ============================================================================

print("\n" + "="*60)
print("Critères de stabilité pour l'intégration temporelle...")
print("="*60)

# Nombre de Courant-Friedrichs-Lewy (CFL)
u_max = np.max(np.abs(problem.u))
v_max = np.max(np.abs(problem.v))
V_max = np.sqrt(u_max**2 + v_max**2)

print(f"\nVitesse maximale: |V|_max = {V_max:.4f} m/s")
print(f"  u_max = {u_max:.4f}, v_max = {v_max:.4f}")

# Pas de temps critique pour Euler explicite
dt_convection = 0.5 * min(problem.dx / u_max, problem.dy / v_max) if u_max > 0 else np.inf
dt_diffusion = 0.25 * problem.rho * min(problem.dx**2, problem.dy**2) / problem.Gamma

print(f"\nPas de temps critiques (Euler explicite):")
print(f"  Convection: Δt < {dt_convection:.6f} s")
print(f"  Diffusion: Δt < {dt_diffusion:.6f} s")
print(f"  Recommandé: Δt < {min(dt_convection, dt_diffusion):.6f} s")

# Nombre de Péclet
Pe_x = problem.rho * u_max * problem.dx / problem.Gamma
Pe_y = problem.rho * v_max * problem.dy / problem.Gamma

print(f"\nNombres de Péclet:")
print(f"  Pe_x = {Pe_x:.4f}")
print(f"  Pe_y = {Pe_y:.4f}")
print(f"  Pe_max = {max(Pe_x, Pe_y):.4f}")

if max(Pe_x, Pe_y) < 2:
    print("  → Schéma centré stable et recommandé")
else:
    print("  → Schéma upwind recommandé (centré peut osciller)")

# ============================================================================
# RÉCAPITULATIF
# ============================================================================

print("\n" + "="*60)
print("RÉCAPITULATIF")
print("="*60)

print("\nOpérateurs spatiaux construits:")
print(f"  ✓ Matrice L ({L.shape[0]}×{L.shape[1]})")
print(f"  ✓ Vecteur b ({b.shape[0]})")

print("\nPour résoudre l'équation instationnaire:")
print(f"  1. Condition initiale: φ(x,y,0) = 0")
print(f"  2. Intégrer: dφ/dt = -1/ρ * (L·φ + b)")
print(f"  3. De t=0 à t=0.12 s")
print(f"  4. Calculer le flux final à t=0.12 s")

print("\n" + "="*60)
print("TERMINÉ!")
print("="*60)
