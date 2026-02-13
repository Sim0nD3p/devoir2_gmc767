"""
Discrétisation spatiale par différences finies de l'équation de transport 2D
au voisinage d'un point de stagnation

Équation: ρ ∂φ/∂t + ρ U·∇φ = Γ ∇²φ
où U·∇φ = u ∂φ/∂x + v ∂φ/∂y
et ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²

Champ de vitesse: u = x, v = -y (écoulement au point de stagnation)

Discrétisation sans la partie temporelle:
ρ U·∇φ - Γ ∇²φ = RHS (qui sera utilisé pour l'intégration temporelle)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, diags, csr_matrix
from scipy.sparse.linalg import spsolve

class TransportDifferencesFinies2D:
    def __init__(self, Lx=1.0, Ly=1.0, nx=41, ny=41, rho=1.2, Gamma=0.1):
        """
        Initialisation du problème avec différences finies
        
        Paramètres:
        -----------
        Lx, Ly : float
            Dimensions du domaine [0, Lx] × [0, Ly]
        nx, ny : int
            Nombre de points de grille en x et y
        rho : float
            Densité du fluide (kg/m³)
        Gamma : float
            Coefficient de diffusion (m²/s)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.rho = rho
        self.Gamma = Gamma
        
        # Création de la grille uniforme
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Champ de vitesse au point de stagnation
        self.u = self.X      # u = x
        self.v = -self.Y     # v = -y
        
        # Initialisation du champ φ (condition initiale: φ = 0 partout)
        self.phi = np.zeros((ny, nx))
        
        # Nombre de points intérieurs (hors frontières)
        self.n_interior = (nx - 2) * (ny - 2)
        
        print(f"Grille créée: {nx}×{ny} points")
        print(f"Pas d'espace: Δx = {self.dx:.5f}, Δy = {self.dy:.5f}")
        print(f"Points intérieurs: {self.n_interior}")
        
    def get_index(self, i, j):
        """
        Convertit les indices (i,j) 2D en indice 1D pour les points intérieurs
        
        Convention: i = indice en y (lignes), j = indice en x (colonnes)
        Points intérieurs: 1 ≤ i ≤ ny-2, 1 ≤ j ≤ nx-2
        """
        if i < 1 or i > self.ny - 2 or j < 1 or j > self.nx - 2:
            return None
        return (i - 1) * (self.nx - 2) + (j - 1)
    
    def get_ij(self, idx):
        """
        Convertit l'indice 1D en indices (i,j) 2D
        """
        i = idx // (self.nx - 2) + 1
        j = idx % (self.nx - 2) + 1
        return i, j
    
    def build_convection_operator(self, scheme='upwind'):
        """
        Construit l'opérateur de convection: ρ U·∇φ
        
        Schémas disponibles:
        - 'upwind': Décentré amont (stable, diffusif)
        - 'central': Centré (ordre 2, peut osciller)
        
        Retourne:
        ---------
        C : matrice sparse (n_interior × n_interior)
            Opérateur de convection
        """
        n = self.n_interior
        C = lil_matrix((n, n))
        
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                idx = self.get_index(i, j)
                
                # Vitesses au point (i,j)
                u_ij = self.u[i, j]
                v_ij = self.v[i, j]
                
                if scheme == 'upwind':
                    # Schéma décentré amont pour ∂φ/∂x
                    if u_ij > 0:
                        # Décentré arrière: (φ_ij - φ_{i,j-1}) / dx
                        coef_x_P = u_ij / self.dx
                        coef_x_W = -u_ij / self.dx
                        coef_x_E = 0
                    else:
                        # Décentré avant: (φ_{i,j+1} - φ_ij) / dx
                        coef_x_P = -u_ij / self.dx
                        coef_x_W = 0
                        coef_x_E = u_ij / self.dx
                    
                    # Schéma décentré amont pour ∂φ/∂y
                    if v_ij > 0:
                        # Décentré arrière: (φ_ij - φ_{i-1,j}) / dy
                        coef_y_P = v_ij / self.dy
                        coef_y_S = -v_ij / self.dy
                        coef_y_N = 0
                    else:
                        # Décentré avant: (φ_{i+1,j} - φ_ij) / dy
                        coef_y_P = -v_ij / self.dy
                        coef_y_S = 0
                        coef_y_N = v_ij / self.dy
                    
                    # Coefficient diagonal
                    C[idx, idx] = self.rho * (coef_x_P + coef_y_P)
                    
                    # Voisin ouest (j-1)
                    if j > 1:
                        idx_W = self.get_index(i, j-1)
                        C[idx, idx_W] = self.rho * coef_x_W
                    
                    # Voisin est (j+1)
                    if j < self.nx - 2:
                        idx_E = self.get_index(i, j+1)
                        C[idx, idx_E] = self.rho * coef_x_E
                    
                    # Voisin sud (i-1)
                    if i > 1:
                        idx_S = self.get_index(i-1, j)
                        C[idx, idx_S] = self.rho * coef_y_S
                    
                    # Voisin nord (i+1)
                    if i < self.ny - 2:
                        idx_N = self.get_index(i+1, j)
                        C[idx, idx_N] = self.rho * coef_y_N
                
                elif scheme == 'central':
                    # Schéma centré pour ∂φ/∂x: u (φ_{i,j+1} - φ_{i,j-1}) / (2dx)
                    coef_x = u_ij / (2 * self.dx)
                    
                    # Schéma centré pour ∂φ/∂y: v (φ_{i+1,j} - φ_{i-1,j}) / (2dy)
                    coef_y = v_ij / (2 * self.dy)
                    
                    # Voisin ouest (j-1)
                    if j > 1:
                        idx_W = self.get_index(i, j-1)
                        C[idx, idx_W] = -self.rho * coef_x
                    
                    # Voisin est (j+1)
                    if j < self.nx - 2:
                        idx_E = self.get_index(i, j+1)
                        C[idx, idx_E] = self.rho * coef_x
                    
                    # Voisin sud (i-1)
                    if i > 1:
                        idx_S = self.get_index(i-1, j)
                        C[idx, idx_S] = -self.rho * coef_y
                    
                    # Voisin nord (i+1)
                    if i < self.ny - 2:
                        idx_N = self.get_index(i+1, j)
                        C[idx, idx_N] = self.rho * coef_y
        
        return C.t0ocsr()
    
    def build_diffusion_operator(self):
        """
        Construit l'opérateur de diffusion: Γ ∇²φ
        
        Utilise des différences finies centrées d'ordre 2:
        ∂²φ/∂x² ≈ (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1}) / dx²
        ∂²φ/∂y² ≈ (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j}) / dy²
        
        Retourne:
        ---------
        D : matrice sparse (n_interior × n_interior)
            Opérateur de diffusion
        """
        n = self.n_interior
        D = lil_matrix((n, n))
        
        # Coefficients du laplacien
        alpha = self.Gamma / (self.dx ** 2)
        beta = self.Gamma / (self.dy ** 2)
        
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                idx = self.get_index(i, j)
                
                # Coefficient diagonal: -2α - 2β
                D[idx, idx] = -2 * alpha - 2 * beta
                
                # Voisin ouest (j-1)
                if j > 1:
                    idx_W = self.get_index(i, j-1)
                    D[idx, idx_W] = alpha
                
                # Voisin est (j+1)
                if j < self.nx - 2:
                    idx_E = self.get_index(i, j+1)
                    D[idx, idx_E] = alpha
                
                # Voisin sud (i-1)
                if i > 1:
                    idx_S = self.get_index(i-1, j)
                    D[idx, idx_S] = beta
                
                # Voisin nord (i+1)
                if i < self.ny - 2:
                    idx_N = self.get_index(i+1, j)
                    D[idx, idx_N] = beta
        
        return D.tocsr()
    
    def build_boundary_source(self):
        """
        Construit le vecteur source dû aux conditions aux limites
        
        Retourne:
        ---------
        b : vecteur (n_interior,)
            Contribution des conditions aux limites
        """
        b = np.zeros(self.n_interior)
        
        alpha = self.Gamma / (self.dx ** 2)
        beta = self.Gamma / (self.dy ** 2)
        
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                idx = self.get_index(i, j)
                
                # Contribution de la frontière ouest (j=0)
                if j == 1:
                    b[idx] -= alpha * self.phi[i, 0]
                
                # Contribution de la frontière est (j=nx-1)
                if j == self.nx - 2:
                    b[idx] -= alpha * self.phi[i, self.nx - 1]
                
                # Contribution de la frontière sud (i=0)
                if i == 1:
                    b[idx] -= beta * self.phi[0, j]
                
                # Contribution de la frontière nord (i=ny-1)
                if i == self.ny - 2:
                    b[idx] -= beta * self.phi[self.ny - 1, j]
        
        return b
    
    def build_spatial_operator(self, scheme='upwind'):
        """
        Construit l'opérateur spatial complet pour l'équation de transport
        
        L(φ) = C(φ) - D(φ) + b
        où C est la convection, D la diffusion, b les conditions aux limites
        
        Pour l'intégration temporelle: dφ/dt = -1/ρ * L(φ)
        
        Retourne:
        ---------
        L : matrice sparse (n_interior × n_interior)
            Opérateur spatial complet
        b : vecteur (n_interior,)
            Terme source des conditions aux limites
        """
        print(f"\nConstruction de l'opérateur spatial (schéma: {scheme})...")
        
        # Opérateur de convection
        C = self.build_convection_operator(scheme=scheme)
        print(f"  Opérateur de convection construit: {C.shape}")
        
        # Opérateur de diffusion
        D = self.build_diffusion_operator()
        print(f"  Opérateur de diffusion construit: {D.shape}")
        
        # Opérateur spatial complet: L = C - D
        L = C - D
        
        # Vecteur source des conditions aux limites
        b = self.build_boundary_source()
        print(f"  Vecteur source construit: {b.shape}")
        
        print(f"  Éléments non-nuls de L: {L.nnz}")
        print(f"  Densité de la matrice: {L.nnz / (L.shape[0]**2) * 100:.2f}%")
        
        return L, b
    
    def set_boundary_conditions(self, west=None, east=None, south=None, north=None):
        """
        Définit les conditions aux limites de Dirichlet
        
        Paramètres:
        -----------
        west, east, south, north : float, array ou None
            Valeurs aux frontières. Si None, la valeur reste inchangée
        """
        if west is not None:
            if np.isscalar(west):
                self.phi[:, 0] = west
            else:
                self.phi[:, 0] = west
        
        if east is not None:
            if np.isscalar(east):
                self.phi[:, -1] = east
            else:
                self.phi[:, -1] = east
        
        if south is not None:
            if np.isscalar(south):
                self.phi[0, :] = south
            else:
                self.phi[0, :] = south
        
        if north is not None:
            if np.isscalar(north):
                self.phi[-1, :] = north
            else:
                self.phi[-1, :] = north
    
    def compute_diffusive_flux_west(self):
        """
        Calcule le flux diffusif à travers la paroi ouest
        
        Flux diffusif = -Γ ∂φ/∂x|_{x=0}
        
        Utilise une différence finie décentrée avant d'ordre 2:
        ∂φ/∂x|_{x=0} ≈ (-3φ_0 + 4φ_1 - φ_2) / (2Δx)
        
        Retourne:
        ---------
        flux_distribution : array (ny,)
            Distribution du flux le long de y
        flux_total : float
            Flux total intégré (W/m en 2D)
        flux_average : float
            Flux moyen par unité de longueur (W/m²)
        """
        # Différence finie décentrée avant d'ordre 2
        dphidx_west = (-3*self.phi[:, 0] + 4*self.phi[:, 1] - self.phi[:, 2]) / (2*self.dx)
        
        # Flux diffusif (convention: positif vers l'extérieur du domaine)
        flux_distribution = -self.Gamma * dphidx_west
        
        # Intégration le long de la frontière ouest (méthode des trapèzes)
        from scipy.integrate import trapezoid
        flux_total = trapezoid(flux_distribution, self.y)
        
        # Flux moyen par unité de surface
        flux_average = flux_total / self.Ly
        
        return flux_distribution, flux_total, flux_average
    
    def compute_rhs(self, phi_interior, L, b):
        """
        Calcule le membre de droite pour l'intégration temporelle
        
        dφ/dt = RHS = -1/ρ * (L·φ + b)
        
        Paramètres:
        -----------
        phi_interior : array (n_interior,)
            Valeurs de φ aux points intérieurs (vecteur 1D)
        L : matrice sparse
            Opérateur spatial
        b : array (n_interior,)
            Terme source des conditions aux limites
        
        Retourne:
        ---------
        rhs : array (n_interior,)
            Membre de droite pour l'intégration temporelle
        """
        rhs = -1.0 / self.rho * (L @ phi_interior + b)
        return rhs
    
    def phi_to_interior(self):
        """
        Extrait les valeurs intérieures de φ dans un vecteur 1D
        """
        phi_interior = np.zeros(self.n_interior)
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                idx = self.get_index(i, j)
                phi_interior[idx] = self.phi[i, j]
        return phi_interior
    
    def interior_to_phi(self, phi_interior):
        """
        Remplit les valeurs intérieures de φ à partir d'un vecteur 1D
        """
        for idx in range(self.n_interior):
            i, j = self.get_ij(idx)
            self.phi[i, j] = phi_interior[idx]
    
    def plot_solution(self, title="Solution φ", save_path=None):
        """
        Visualise la solution et le champ de vitesse
        """
        fig = plt.figure(figsize=(16, 6))
        
        # Sous-graphique 1: Contours de φ
        ax1 = plt.subplot(131)
        contour = ax1.contourf(self.X, self.Y, self.phi, levels=20, cmap='coolwarm')
        ax1.contour(self.X, self.Y, self.phi, levels=10, colors='k', linewidths=0.5, alpha=0.3)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.set_aspect('equal')
        plt.colorbar(contour, ax=ax1, label='φ')
        
        # Sous-graphique 2: Champ de vitesse
        ax2 = plt.subplot(132)
        skip = max(1, self.nx // 20)
        magnitude = np.sqrt(self.u**2 + self.v**2)
        quiver = ax2.quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip],
                           self.u[::skip, ::skip], self.v[::skip, ::skip],
                           magnitude[::skip, ::skip], cmap='viridis')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title('Champ de vitesse (u=x, v=-y)', fontsize=14)
        ax2.set_aspect('equal')
        plt.colorbar(quiver, ax=ax2, label='|V|')
        
        # Sous-graphique 3: Profils de φ
        ax3 = plt.subplot(133)
        # Profil le long de x au milieu (y = Ly/2)
        j_mid = self.ny // 2
        ax3.plot(self.x, self.phi[j_mid, :], 'b-', linewidth=2, label=f'φ(x, y={self.y[j_mid]:.2f})')
        # Profil le long de y au milieu (x = Lx/2)
        i_mid = self.nx // 2
        ax3.plot(self.y, self.phi[:, i_mid], 'r--', linewidth=2, label=f'φ(x={self.x[i_mid]:.2f}, y)')
        ax3.set_xlabel('x ou y', fontsize=12)
        ax3.set_ylabel('φ', fontsize=12)
        ax3.set_title('Profils de φ', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        return fig
    
    def plot_flux_distribution(self, flux_distribution, save_path=None):
        """
        Visualise la distribution du flux à la paroi ouest
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.y, flux_distribution, 'b-', linewidth=2, marker='o', 
                markersize=4, markevery=max(1, self.ny//20))
        ax.set_xlabel('y', fontsize=12)
        ax.set_ylabel('Flux diffusif [W/m² ou unités équivalentes]', fontsize=12)
        ax.set_title('Distribution du flux diffusif à la paroi ouest (x=0)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # Ajout de statistiques
        flux_max = np.max(flux_distribution)
        flux_min = np.min(flux_distribution)
        y_max = self.y[np.argmax(flux_distribution)]
        y_min = self.y[np.argmin(flux_distribution)]
        
        textstr = f'Max: {flux_max:.4f} à y={y_max:.3f}\nMin: {flux_min:.4f} à y={y_min:.3f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        return fig
    
    def print_summary(self):
        """
        Affiche un résumé des paramètres et de l'état actuel
        """
        print("\n" + "="*70)
        print("RÉSUMÉ DU PROBLÈME")
        print("="*70)
        print(f"Domaine: [{0}, {self.Lx}] × [{0}, {self.Ly}]")
        print(f"Grille: {self.nx} × {self.ny} points")
        print(f"Résolution: Δx = {self.dx:.5f}, Δy = {self.dy:.5f}")
        print(f"Propriétés du fluide: ρ = {self.rho}, Γ = {self.Gamma}")
        print(f"Points intérieurs: {self.n_interior}")
        print(f"\nChamp de vitesse: u = x, v = -y")
        print(f"φ min = {self.phi.min():.6f}, φ max = {self.phi.max():.6f}")
        print("="*70)


def main():
    """
    Programme principal de démonstration
    """
    print("="*70)
    print("DISCRÉTISATION SPATIALE PAR DIFFÉRENCES FINIES")
    print("Équation de transport 2D au voisinage d'un point de stagnation")
    print("="*70)
    
    # Création du problème
    problem = TransportDifferencesFinies2D(
        Lx=1.0,      # Domaine: [0, 1] × [0, 1]
        Ly=1.0,
        nx=41,       # Grille 41×41
        ny=41,
        rho=1.2,     # Densité
        Gamma=0.1    # Coefficient de diffusion
    )
    
    # Affichage du résumé
    problem.print_summary()
    
    # Construction de l'opérateur spatial avec schéma upwind
    print("\n" + "-"*70)
    print("CONSTRUCTION DE L'OPÉRATEUR SPATIAL")
    print("-"*70)
    L_upwind, b = problem.build_spatial_operator(scheme='upwind')
    
    # Application des conditions aux limites
    print("\n" + "-"*70)
    print("CONDITIONS AUX LIMITES")
    print("-"*70)
    print("Ouest (x=0): φ = 1.0")
    print("Est (x=Lx): φ = 0.0")
    print("Sud (y=0): φ = 0.0")
    print("Nord (y=Ly): φ = 0.0")
    problem.set_boundary_conditions(west=1.0, east=0.0, south=0.0, north=0.0)
    
    # Recalcul du vecteur source avec les nouvelles conditions
    b = problem.build_boundary_source()
    
    # Visualisation de la structure de la matrice
    print("\n" + "-"*70)
    print("VISUALISATION DE LA STRUCTURE DE LA MATRICE")
    print("-"*70)
    
    fig_matrix = plt.figure(figsize=(10, 10))
    plt.spy(L_upwind, markersize=1, color='blue')
    plt.title('Structure de la matrice de l\'opérateur spatial L', fontsize=14)
    plt.xlabel('Colonne', fontsize=12)
    plt.ylabel('Ligne', fontsize=12)
    plt.savefig('/home/claude/matrix_structure.png', dpi=150, bbox_inches='tight')
    print("✓ Structure de la matrice sauvegardée: matrix_structure.png")
    
    # Test avec un état stationnaire (résolution de L·φ = -b)
    print("\n" + "-"*70)
    print("TEST: RÉSOLUTION DU PROBLÈME STATIONNAIRE")
    print("-"*70)
    print("Résolution de: ρ U·∇φ = Γ ∇²φ + b")
    print("(équivalent à L·φ = -b)")
    
    # Résolution du système
    phi_interior_steady = spsolve(L_upwind, -b)
    
    # Mise à jour de φ
    problem.interior_to_phi(phi_interior_steady)
    
    print(f"✓ Solution stationnaire calculée")
    print(f"  φ min = {problem.phi.min():.6f}")
    print(f"  φ max = {problem.phi.max():.6f}")
    
    # Calcul du flux à la paroi ouest
    print("\n" + "-"*70)
    print("CALCUL DU FLUX DIFFUSIF À LA PAROI OUEST")
    print("-"*70)
    
    flux_dist, flux_total, flux_avg = problem.compute_diffusive_flux_west()
    
    print(f"Flux total: {flux_total:.6f}")
    print(f"Flux moyen par unité de longueur: {flux_avg:.6f}")
    print(f"Flux maximum: {flux_dist.max():.6f} à y = {problem.y[np.argmax(flux_dist)]:.3f}")
    print(f"Flux minimum: {flux_dist.min():.6f} à y = {problem.y[np.argmin(flux_dist)]:.3f}")
    
    # Visualisations
    print("\n" + "-"*70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("-"*70)
    
    problem.plot_solution(
        title="Solution stationnaire φ",
        save_path='/home/claude/solution_stationnaire.png'
    )
    
    problem.plot_flux_distribution(
        flux_dist,
        save_path='/home/claude/flux_distribution.png'
    )
    
    print("\n" + "="*70)
    print("DISCRÉTISATION SPATIALE TERMINÉE!")
    print("="*70)
    print("\nLes opérateurs spatiaux sont prêts pour l'intégration temporelle:")
    print("  - Matrice L (opérateur spatial complet)")
    print("  - Vecteur b (conditions aux limites)")
    print("\nPour l'intégration temporelle, utiliser:")
    print("  dφ/dt = -1/ρ * (L·φ + b)")
    print("="*70)
    
    return problem, L_upwind, b


if __name__ == "__main__":
    problem, L, b = main()
    plt.show()
