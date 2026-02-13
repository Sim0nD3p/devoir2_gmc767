import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix

class FiniteDifferenceMethod:
    def __init__(self, Lx, Ly, nx, ny, rho, gamma):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.rho = rho
        self.gamma = gamma

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        
        # Convention: shape = (ny, nx) -> (Lignes/Y, Colonnes/X)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialisation du champ phi
        self.phi = np.zeros((ny, nx))
        
        # Vitesses
        self.u = self.X
        self.v = -self.Y
        
        self.n_total = self.nx * self.ny

    def get_index(self, i, j):
        # i = ligne (y), j = colonne (x)
        return i * self.nx + j

    def apply_bcs(self):
        """Met à jour self.phi avec les conditions limites"""
        
        # 1. OUEST (Mur, Dirichlet variable): x=0, col 0
        # phi = 1 - y
        self.phi[:, 0] = 1.0 - self.y

        # 2. NORD (Inlet, Dirichlet): y=Ly, ligne -1
        # phi = 0
        self.phi[-1, :] = 0.0

        # 3. SUD (Symétrie, Neumann): y=0, ligne 0
        # phi_0 = (4*phi_1 - phi_2) / 3
        self.phi[0, :] = (4.0 * self.phi[1, :] - 1.0 * self.phi[2, :]) / 3.0

        # 4. EST (Outlet, Neumann): x=Lx, col -1
        # phi_N = (4*phi_N-1 - phi_N-2) / 3
        self.phi[:, -1] = (4.0 * self.phi[:, -2] - 1.0 * self.phi[:, -3]) / 3.0
        
        # Coin Sud-Ouest (Conflit Mur/Symétrie -> Mur gagne)
        self.phi[0, 0] = 1.0

    def build_matrices(self, scheme='CD2'):
        """Construit les matrices C et D une seule fois"""
        print('Construction des matrices C et D...')
        n = self.n_total 
        C = lil_matrix((n, n))
        D = lil_matrix((n, n))
        
        # Coefficients pré-calculés
        # Attention: on divise tout par rho à la fin, ou ici. 
        # Pour RHS = -U.grad(phi) + (Gamma/rho)*Laplace(phi)
        
        # Coeffs Convection (divisés par rho plus tard ou intégrés ici?)
        # Convention standard: dphi/dt = ...
        # Donc on met 1/rho devant tout ou on l'intègre.
        # Ici je garde tes coeffs bruts, on divisera par rho dans get_rhs
        
        coef_diff = self.gamma # On divisera par rho dans le RHS
        
        # On boucle sur l'INTÉRIEUR seulement (1 à N-1)
        for i in range(1, self.ny - 1):       # Lignes (Y)
            for j in range(1, self.nx - 1):   # Colonnes (X)
                
                idx = self.get_index(i, j)
                u_ij = self.u[i, j]
                v_ij = self.v[i, j]

                # --- CONVECTION ---
                # CD2: (phi_E - phi_W)/2dx
                c_x = (self.rho * u_ij) / (2 * self.dx)
                c_y = (self.rho * v_ij) / (2 * self.dy)
                
                # Voisins
                idx_E = self.get_index(i, j+1)
                idx_W = self.get_index(i, j-1)
                idx_N = self.get_index(i+1, j)
                idx_S = self.get_index(i-1, j)
                
                # Remplissage C
                C[idx, idx_E] += c_x
                C[idx, idx_W] -= c_x
                C[idx, idx_N] += c_y
                C[idx, idx_S] -= c_y

                # --- DIFFUSION ---
                # CD2: (phi_E - 2phi + phi_W)/dx2
                d_x = coef_diff / (self.dx**2)
                d_y = coef_diff / (self.dy**2)
                
                D[idx, idx] -= 2*(d_x + d_y)
                D[idx, idx_E] += d_x
                D[idx, idx_W] += d_x
                D[idx, idx_N] += d_y
                D[idx, idx_S] += d_y

        return C.tocsr(), D.tocsr()

    def get_rhs(self, C, D):
        """
        Calcule R(phi) = 1/rho * (Diff - Conv)
        """
        # 1. Mise à jour des BCs sur le champ actuel
        self.apply_bcs()
        
        # 2. Aplatir phi pour le calcul matriciel
        phi_flat = self.phi.flatten()
        
        # 3. Calcul: Force = Diffusion - Convection
        # RHS = (1/rho) * (D*phi - C*phi)
        force = (D - C).dot(phi_flat)
        rhs = force / self.rho
        
        # 4. On remet en 2D pour affichage ou futur pas de temps
        return rhs.reshape((self.ny, self.nx))

# --- TEST ---
FDM = FiniteDifferenceMethod(Lx=1, Ly=1, nx=11, ny=11, rho=1.2, gamma=0.1)

# 1. Construire les matrices (une seule fois)
C, D = FDM.build_matrices()

# 2. Calculer le RHS (simule un pas de temps)
rhs = FDM.get_rhs(C, D)

print(f"RHS shape: {rhs.shape}")
print("Valeur max du RHS:", np.max(np.abs(rhs)))

# Visualisation pour vérifier que les BCs sont bien appliqués
plt.imshow(FDM.phi, origin='lower')
plt.colorbar(label='Phi')
plt.title("Champ Phi avec Conditions Limites")
plt.show()