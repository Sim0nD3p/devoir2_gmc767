import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, diags, csr_matrix, eye
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation

class FiniteDifferenceMethod:
    def __init__(self, Lx, Ly, nx, ny, rho, gamma):
        self.time_scheme = ''
        self.Lx = Lx
        self.Ly = Ly

        self.nx = nx
        self.ny = ny

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        self.rho = rho
        self.gamma = gamma

        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.phi = np.zeros((nx, ny))


        self.u = self.X
        self.v = -self.Y

        self.t_end = 0.5
        self.dt = 0.001
        self.time_steps = []
        self.phi_solution_time = []
        
        self.n_total = self.nx * self.ny

        plt.ion()


    def clear_solution(self):
        self.phi_solution_time = []
        self.time_steps = []
        self.phi = np.zeros((self.nx, self.ny))

    def get_index(self, i, j):
        return i * self.ny + j
    
    def get_ij(self, idx):
        """
        Not implemented anywhere
        
        :param idx: Storage index
        """
        i = idx // self.ny
        j = idx % self.ny
        
        return i, j

    def build_convection_term(self, scheme='CD2'):
        """
        Building convection term for internal nodes
        
        :param scheme: Convection scheme (CD2 is implemented)
        """

        print('Building convection term')
        n = self.n_total 
        C = lil_matrix((n, n))
        
        # Loop au travers les noeuds internes en appliquant les coefficient de discretisation aux points P et voisins N-S-W-E
        # Voir section 3.8, fig 3.6 of Ferziger et al., "Computational Methods for Fluid Dynamics", 4th Ed.
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                    idx = self.get_index(i, j)

                    u_ij = self.u[i, j]
                    v_ij = self.v[i, j]

                    if scheme == 'CD2':
                        coeff_x = u_ij / (2*self.dx)
                        coeff_y = v_ij / (2*self.dy)

                        if i > 0 and i < self.nx-1:     
                            idx_W = self.get_index(i-1, j)
                            idx_E = self.get_index(i+1, j)
                            C[idx, idx_W] = -coeff_x
                            C[idx, idx_E] = coeff_x
                        
                        if j > 0 and j < self.ny-1:
                            idx_S = self.get_index(i, j-1)
                            idx_N = self.get_index(i, j+1)
                            C[idx, idx_S] = -coeff_y
                            C[idx, idx_N] = coeff_y

        return C.tocsr()
    
    def build_diffusive_term(self, scheme='CD2'):
        n = self.n_total
        D = lil_matrix((n, n))

        base_coeff = self.gamma / self.rho

        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                    idx = self.get_index(i, j)

                    D[idx, idx] = -2 * base_coeff * (1/(self.dx)**2 + 1/(self.dy)**2)

                    coeff_x = base_coeff / (self.dx)**2
                    if i > 0 and i < self.nx-1:
                        idx_E = self.get_index(i+1, j)
                        idx_W = self.get_index(i-1, j)

                        D[idx, idx_E] = coeff_x
                        D[idx, idx_W] = coeff_x

                    coeff_y = base_coeff / (self.dy)**2
                    if j > 0 and j < self.ny-1:
                        idx_S = self.get_index(i, j-1)
                        idx_N = self.get_index(i, j+1)

                        D[idx, idx_S] = coeff_y
                        D[idx, idx_N] = coeff_y
        
        return D.tocsr()
    
    def apply_boundary_conditions(self):
        """
        Ici, on applique les conditions limites directement sur le champ scalaire phi, à chaque itération explicite, pour faire en sorte qu'elles sont respectées
        
        Conventions (indexing='ij'):
        - i (1er indice) = x
        - j (2e indice)  = y
        """
        
        # MUR SUD (y=0, j=0) -> NEUMANN (Dérivée nulle en y)
        # Formule ordre 2 : phi_0 = (4*phi_1 - phi_2) / 3
        # On modifie la ligne du bas (j=0) en utilisant les lignes j=1 et j=2
        self.phi[:, 0] = (4.0 * self.phi[:, 1] - 1.0 * self.phi[:, 2]) / 3.0

        # MUR EST (x=Lx, i=-1) -> NEUMANN (Dérivée nulle en x)
        # Formule ordre 2 : phi_N = (4*phi_N-1 - phi_N-2) / 3
        # On modifie la colonne de droite (i=-1) en utilisant i=-2 et i=-3
        self.phi[-1, :] = (4.0 * self.phi[-2, :] - 1.0 * self.phi[-3, :]) / 3.0

        # MUR OUEST (x=0, i=0) -> DIRICHLET
        # Condition : phi = 1 - y
        # On applique sur toute la colonne de gauche
        self.phi[0, :] = 1.0 - self.y

        # MUR NORD (y=Ly, j=-1) -> DIRICHLET
        # Condition : phi = 0
        # On applique sur toute la ligne du haut
        self.phi[:, -1] = 0.0
    

    def solve_steady_state(self):
        # 1. Construction des matrices physiques (D - C)
        self.time_scheme = 'steady-state'
        D = self.build_diffusive_term()
        C = self.build_convection_term()
        A = (D - C).tolil()
        b = np.zeros(self.n_total)
        
        A = self.replace_rows_A_matrix(A)
        b = self.replace_b(b)
        # Résolution
        phi_flat = spsolve(A, b)
        self.phi.flat[:] = phi_flat
        return self.phi


    def solve_explicit(self, plot_every=10):
        # 1. Matrices physiques (ne changent pas)
        # Note: build_matrices ne doit calculer que l'intérieur, 
        # ou alors on ignore ce qu'elle met sur les bords.
        self.time_scheme = 'explicit'
        dt = self.dt
        t_end = self.t_end
        D = self.build_diffusive_term()
        C = self.build_convection_term()
        if plot_every != False: self.init_live_plot()
        n_steps = int(t_end / dt)
        print(f"Début Euler Explicite : {n_steps} itérations")

        for n in range(n_steps):
            # A. Calcul de la variation (RHS)
            # rhs = 1/rho * (Diff - Conv)
            phi_flat = self.phi.flatten() # Attention à l'ordre si tu utilises 'F'
            rhs = (D - C).dot(phi_flat)
            
            # B. Mise à jour de l'état (Euler)
            # phi(t+1) = phi(t) + dt * rhs
            self.phi += dt * rhs.reshape((self.nx, self.ny)) # reshape suit indexing='ij' par défaut

            # C. APPLICATION DES CONDITIONS LIMITES
            # C'est ici qu'on "écrase" les valeurs fausses aux bords
            self.apply_boundary_conditions()

            
            #self.phi_solution_time.append(self.phi.copy())
            if n % 50 == 0:

                self.phi_solution_time.append(self.phi.copy())
                self.time_steps.append(n * dt)
            if plot_every != False:
                if n % plot_every == 0:
                    self.update_live_plot(n, n * dt)

        plt.show() # Garde la fenêtre ouverte    
        print("Calcul terminé.")
        return self.phi
    
    def solve_implicit(self, plot_every=5):
        """
        Résout l'équation de transport avec le schéma Euler Implicite.
        Équation : [I - (dt/rho)*(D - C)] * phi^{n+1} = phi^n
        """
        dt = self.dt
        t_end = self.t_end
        if plot_every != False: self.init_live_plot()
        self.time_scheme = 'implicit'

        self.phi_solution_time = []
        self.time_steps = []

        # 1. Construction des matrices physiques brutes
        C = self.build_convection_term()
        D = self.build_diffusive_term()
        
        
        
        # 2. Construction de l'opérateur de gauche (LHS) : A = I - (dt/rho)*(D - C)
        # On utilise le format 'lil' pour permettre les modifications de lignes (Row Replacement)
        I = eye(self.n_total, format='lil')
        A = I - dt * (D - C)
        A = A.tolil()

        A_csr = self.replace_rows_A_matrix(A)

        # 3. BOUCLE TEMPORELLE
        n_steps = int(t_end / dt)
        print(f"Début de la simulation : {n_steps} pas de temps.")

        for n in range(n_steps):
            # Le vecteur b de base est la solution actuelle phi^n
            b = self.phi.flatten()
            b = self.replace_b(b)
            # 4. RÉSOLUTION Ax = b
            
            phi_new = spsolve(A_csr, b)
            
            # Mise à jour du champ phi pour le prochain tour
            # self.phi.flat[:] doesnt copy phi, modify a view of the original table
            self.phi.flat[:] = phi_new

            
            self.phi_solution_time.append(self.phi.copy())
            self.time_steps.append(n * dt)
            if plot_every != False:
                if n % plot_every == 0:
                    self.update_live_plot(n, n * dt)
            
        plt.show() # Garde la fenêtre ouverte


        return self.phi
    
    def get_temperature_flux(self, plot=True):
        dx = self.dx
        gamma = self.gamma
        x_i = 0
        f_t = []

        for i in range(len(self.time_steps)):
            phi = self.phi_solution_time[i]

            # Dérivée 2e ordre backward
            dphi_dx_ti = (-3 * phi[x_i,:] + 4 * phi[x_i+1,:] - phi[x_i+2,:]) / (2*dx)
            # Intégrale le long de y
            integral_y = np.trapz(dphi_dx_ti, self.y)
            f_t.append(-gamma * integral_y)

        
        if plot == True:
            # 1. Création de la figure avec un style épuré
            plt.style.use('seaborn-v0_8-muted') # Style doux et pro
            fig, ax = plt.subplots(figsize=(8, 5))

            # 2. Le tracé avec des marqueurs stylisés
            ax.plot(self.time_steps, f_t, 
                    color='#4C72B0',           # Bleu acier élégant
                    linewidth=2,               # Ligne plus épaisse
                    marker='o',                # Cercles au lieu d'étoiles
                    markersize=4,              # Taille discrète
                    markerfacecolor='white',   # Coeur blanc pour un look "hollow"
                    markeredgewidth=1.5,
                    label='Flux total $f(t)$')

            # 3. Ombrage sous la courbe (le petit plus "cute")
            ax.fill_between(self.time_steps, f_t, color='#4C72B0', alpha=0.1)

            # 4. Habillage (Labels en LaTeX)
            ax.set_title(f'Évolution temporelle du flux thermique total $\Delta t$ = {self.dt}', 
                         fontsize=14, pad=20)
            ax.set_xlabel(r'Temps $t$ [s]', fontsize=12)
            ax.set_ylabel(r'Flux total [W]', fontsize=12)

            # 5. Grille et Bordures (Spines)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.spines['top'].set_visible(False)    # On enlève le cadre du haut
            ax.spines['right'].set_visible(False)  # On enlève le cadre de droite
        
        print(f'Temparature flux is {f_t[-1]}')
        return f_t
    
    def plot_multiple(self, xs, ys, labels, title, x_label, y_label):
        fig, ax = plt.subplots(figsize=(8, 5))

        for i in range(len(xs)):
            ax.plot(xs[i], ys[i], 
                    #color="#0565FF",           # Bleu acier élégant
                    linewidth=1,               # Ligne plus épaisse
                    marker='.',                # Cercles au lieu d'étoiles
                    markersize=1,              # Taille discrète
                    markerfacecolor='white',   # Coeur blanc pour un look "hollow"
                    #markeredgewidth=1.5,

                    label=labels[i])

        ax.set_title(title, fontsize=14, pad=20)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()



    def replace_rows_A_matrix(self, A):
        print("Application du Row Replacement dans la matrice A...")
        # Ici, on cherche a remplacer les coefficient qui, multipliés par le vecteur phi, donnent le vecteur phi_m+1

        # --- A. DIRICHLET (Valeurs imposées) ---
        # OUEST (i=0) : phi = 1 - y
        for j in range(self.ny):
            idx = self.get_index(0, j)
            A.rows[idx] = [] ; A.data[idx] = [] # Efface la ligne
            A[idx, idx] = 1.0                   # Impose 1 sur la diagonale

        # NORD (j=ny-1) : phi = 0
        for i in range(self.nx):
            idx = self.get_index(i, self.ny - 1)
            A.rows[idx] = [] ; A.data[idx] = []
            A[idx, idx] = 1.0

        # --- B. NEUMANN (Gradients imposés : 3*phi0 - 4*phi1 + phi2 = 0) ---
        # SUD (j=0) : Relation avec voisins Nord (j=1, j=2)
        # On exclut les coins déjà traités par Dirichlet (i=0 et j=ny-1)
        for i in range(1, self.nx):
            idx = self.get_index(i, 0)
            idx_N1 = self.get_index(i, 1)
            idx_N2 = self.get_index(i, 2)
            
            A.rows[idx] = [] ; A.data[idx] = []
            A[idx, idx] = 3.0
            A[idx, idx_N1] = -4.0
            A[idx, idx_N2] = 1.0

        # EST (i=nx-1) : Relation avec voisins Ouest (i-1, i-2)
        # On exclut le point déjà traité au Nord (j=ny-1)
        for j in range(self.ny - 1):
            idx = self.get_index(self.nx - 1, j)
            idx_W1 = self.get_index(self.nx - 2, j)
            idx_W2 = self.get_index(self.nx - 3, j)
            
            A.rows[idx] = [] ; A.data[idx] = []
            A[idx, idx] = 3.0
            A[idx, idx_W1] = -4.0
            A[idx, idx_W2] = 1.0

        # Conversion en CSR pour une résolution optimale
        A_csr = A.tocsr()
        return A_csr


    def replace_b(self, b):
        # --- MISE À JOUR DU VECTEUR b POUR LES BORDURES ---
        # Pour les points où on a fait un "Row Replacement", le RHS change :
        
        # Dirichlet Ouest : b[idx] = 1 - y
        for j in range(self.ny):
            idx = self.get_index(0, j)
            b[idx] = 1.0 - self.y[j]
        # Dirichlet Nord : b[idx] = 0
        for i in range(self.nx):
            idx = self.get_index(i, self.ny - 1)
            b[idx] = 0.0
        # Neumann Sud et Est : l'équation est "... = 0"
        # On force b à 0 pour ces lignes
        for i in range(1, self.nx):
            b[self.get_index(i, 0)] = 0.0
        for j in range(self.ny - 1):
            b[self.get_index(self.nx - 1, j)] = 0.0
        
        return b

    

    def init_live_plot(self):
        plt.ion()  # Force le mode interactif
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        
        # Initialisation de l'image (Column-Major / indexing='ij')
        # On utilise imshow pour la rapidité
        self.im = self.ax.imshow(self.phi.T, origin='lower', 
                                 extent=[0, self.Lx, 0, self.Ly], 
                                 cmap='turbo', vmin=0, vmax=1)
        
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.fig.colorbar(self.im, label='$\phi$')
        self.plot_title = self.ax.set_title("Initialisation du calcul...")
        
        # --- LA RÉPARATION ---
        self.fig.show() # Ouvre la fenêtre
        plt.show(block=False) # Dit explicitement de ne pas bloquer
        self.fig.canvas.draw() # Force le premier rendu
    
    def update_live_plot(self, n, current_time):
        
        # Mise à jour des données sans recréer l'objet (très rapide)
        self.im.set_data(self.phi.T)
        self.plot_title.set_text(f"Itération: {n} | Temps: {current_time:.4f} s")
        
        # --- LA RÉPARATION ---
        # On demande au canvas de se redessiner quand il peut
        self.fig.canvas.draw_idle()
        # On traite les événements GUI (bouger la fenêtre, redimensionner) 
        # pendant un temps très court
        self.fig.canvas.flush_events()
        plt.pause(1e-9) # Micro-pause pour laisser le temps au processeur de dessiner

    def animate(self, filename='evolution.gif'):
        # 1. Vérification de sécurité
        if not self.phi_solution_time or len(self.phi_solution_time) == 0:
            print("Erreur: Aucune donnée sauvegardée. Vérifiez que solve_implicit a bien tourné.")
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        
        # On définit les niveaux de contours une seule fois pour la cohérence
        contour_levels = np.linspace(0, 1, 256)
        
        # 2. Initialisation du premier cadre
        cont = ax.contourf(self.X, self.Y, self.phi_solution_time[0], levels=contour_levels, cmap='turbo')
        fig.colorbar(cont, label='$\phi$')
        
        # On crée l'objet texte pour le titre une seule fois
        title_text = ax.set_title(f"Temps: {self.time_steps[0]:.4f} s")

        def update(frame):
            # frame est l'index envoyé par FuncAnimation
            ax.clear() # On nettoie pour éviter la superposition
            
            # Redessiner le champ à l'instant 'frame'
            c = ax.contourf(self.X, self.Y, self.phi_solution_time[frame], levels=contour_levels, cmap='turbo')
            
            # Mise à jour du titre avec le temps correspondant
            ax.set_title(f"Temps: {self.time_steps[frame]:.4f} s")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            return c.collections

        # 3. CRUCIAL : Définir le nombre exact de frames
        # On dit à FuncAnimation de ne pas aller plus loin que la taille de notre liste
        num_frames = len(self.phi_solution_time)
        
        ani = animation.FuncAnimation(
            fig, 
            update, 
            frames=num_frames, # Utilise la taille réelle de la liste
            blit=False,
            repeat=False
        )

        print(f"Sauvegarde de {num_frames} images dans {filename}...")
        
        # Augmenter le fps si tu as beaucoup d'images (ex: 15 ou 20)
        ani.save(filename, writer='pillow', fps=15)
        plt.close(fig) # Ferme la figure pour libérer la mémoire
        print("Animation terminée.")

    def plot_fig(self, save_plot:False, filename='file'):
        fig, ax = plt.subplots(figsize=(7, 6))
        filename = filename + '.png'



        print(solver.phi)
        levels = np.linspace(0, 1, 256)
        cp = ax.contourf(solver.X, solver.Y, solver.phi, levels=levels, cmap='turbo')
        cbar = fig.colorbar(cp, ax=ax)
        cbar.set_label(r'Scalaire $\phi$', fontsize=12)
        if self.time_scheme != 'steady-state':

            title_str = (f"Champ de transport $\phi$ à $t={solver.time_steps[-1]:.2f}$ s\n"
             f"($\Delta t={solver.dt:.6f}$, Schéma: {solver.time_scheme})")
        else:
            title_str = (f"Champ de transport $\phi$ Schéma: {solver.time_scheme})")

        ax.set_title(title_str, fontsize=13, pad=15)
        ax.set_aspect('equal') # Crucial pour que le domaine ne soit pas déformé
        ax.set_xlabel('Position x [m]')
        ax.set_ylabel('Position y [m]')

        if save_plot:
            # dpi=300 ensures high resolution, bbox_inches='tight' fits the labels
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé : {filename}")

        plt.show()






    



solver = FiniteDifferenceMethod(Lx=1, Ly=1, nx=80, ny=80, rho=1.2, gamma=0.1)
solver.t_end = 0.12
solver.dt = 0.000505

c = solver.solve_implicit(plot_every=False)
#c = solver.solve_explicit(plot_every=False)

#solver.plot_fig(save_plot=True)
#solver.get_temperature_flux()

def multiple_explicit():
    xs = []
    ys = []
    labels = []
    title = 'Flux de température pour différent $\Delta t$'
    xlabel = 'Temps [s]'
    ylabel = 'Flux de température [W]'
    fluxes = []

    dts = [0.000475, 0.000505, 0.000540]
    for dt in dts:
        string = f'$\Delta t$ = {dt} s'
        labels.append(string)
        solver.dt = dt
        c = solver.solve_explicit(plot_every=False)
        f_t = solver.get_temperature_flux(plot=False)
        print(f'Flux de température $\Delta t$ = {dt} s, -> {f_t[-1]} W')
        fluxes.append(f_t[-1])
        ys.append(f_t)
        xs.append(solver.time_steps)

        solver.clear_solution()


    solver.plot_multiple(xs, ys, labels, title, xlabel, ylabel)

    print(f'dt = {dts}')
    print(f'flux = {fluxes}')


def multiple_implicit():
    xs = []
    ys = []
    labels = []
    title = 'Flux de température pour différent $\Delta t$'
    xlabel = 'Temps [s]'
    ylabel = 'Flux de température [W]'
    fluxes = []

    dts = [0.012, 0.006, 0.003, 0.0015, 0.00075]
    for dt in dts:
        string = f'$\Delta t$ = {dt} s'
        labels.append(string)
        solver.dt = dt
        c = solver.solve_implicit(plot_every=False)
        #f_t = solver.get_temperature_flux(plot=False)
        #print(f'Flux de température $\Delta t$ = {dt} s, -> {f_t[-1]} W')
        #fluxes.append(f_t[-1])
        #ys.append(f_t)
        #xs.append(solver.time_steps)

        solver.plot_fig(True, str(dt))
        solver.clear_solution()


    #lver.plot_multiple(xs, ys, labels, title, xlabel, ylabel)

    print(f'dt = {dts}')
    print(f'flux = {fluxes}')

multiple_implicit()
#multiple_explicit()
#c = solver.solve_steady_state()
#solver.plot_fig(save_plot=True)
#solver.animate()
plt.ioff()      # Désactive le mode interactif
plt.show()      # Devient bloquant ici, empêchant le script de se fermer
