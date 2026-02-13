# Devoir #2 GMC767

Le code pour le devoir 2 est fait sous forme de classe Python contenant les différents solveurs discrétisé en différence centrée d'ordre 2 (CD2).

### Équation de départ
L'équation de départ est l'équation de transport pour une quantité scalaire générique $\phi$.

$$\rho \frac{\partial \phi}{\partial t} + \rho \vec{v} \cdot \nabla \phi = \Gamma (\nabla^2 \phi)$$

Celle-ci est immédiatement ré-arrangée de manière à isoler le terme temporel:
$$\frac{\partial\phi}{\partial t}=-\vec{U}\cdot\nabla\phi+\frac{\Gamma}{\rho}\left(\nabla^2\phi\right)$$

## Conditions frontières aux parois
Le domaine comprends 4 parois: N (nord), S (sud), E (est) et W (ouest)

### Paroi Nord
La frontière nord est un ***inlet*** où la quantité $\phi$ = 0. C'est une condition de type Dirichlet puisqu'on impose une quantité à l'entrée.
$$\phi = 0$$
### Paroi Sud
La frontière sud est un plan de symmétrie. Le gradient (dérivée) est imposé à la frontière. C'est donc une condition de type Neumann.

$$\frac{\partial \phi}{\partial y} = 0$$

### Paroi Ouest
La paroi ouest est une paroi de type mur où une quantité $\phi$ est imposée. C'est donc une condition de type Dirichlet. Le long du mur, la valeur de $\phi$ varie de la manière suivante:
$$\phi(y)=1 - y$$
### Paroi Est
La paroi est est une frontière de type ***outlet***. Donc une condition de type **Neumann** où le gradient de $\phi$ est imposé de la manière suivante:

$$\frac{\partial \phi}{\partial x} = 0$$




<div align=center>
    <h3>Domaine de calcul</h3>
    <img src=./images/image_domaineInitial.png width=400px>
</div>

## Discrétisation des termes spaciaux pour la convection et diffusion


### Terme convectif
$$C=\vec{U}\cdot\nabla\phi = u \frac{\partial \phi}{\partial x} + v \frac{\partial \phi}{\partial y}$$

En prenant une dimension, on discrétise ainsi: 
$$\frac{\partial\phi}{\partial x}=\frac{\phi_{i+1}-\phi_{i-1}}{2\Delta x}$$

Pour simplifier le code, la discrétisation est traitée ainsi:

$$\frac{u}{2\Delta x}(\phi_{i+1} - \phi_{i-1})$$

### Terme diffusif
# à faire

## Terme résiduel
$$R(\phi^n) = \frac{\partial\phi^n}{\partial t} = -\vec{U}\cdot\nabla\phi^n + \frac{\Gamma}{\rho}(\nabla^2\phi^n)$$

### Terme diffusif

## Algorithme explicite et implicite


### Méthode explicite
Pour la méthode Euler explicite, on a 
$$\phi^{n+1} = \phi^n + \Delta t \cdot R(\phi^n)$$
À chaque pas de temps, le terme $\Delta t \cdot R(\phi^n)$ est additionné au résultat précédent. Ce terme est composé des termes ***C*** et ***D*** qui sont respectivement les opérateurs de convection et diffusion obetnus par les méthodes `build_convection_term()` et `build_diffusive_term()`. Ces termes, sous forme de matrice, représentent les coefficients des points intérieurs multipliant ensuite le vecteur $\phi^n$. Cette méthodologie est conforme à la section 3.8, fig 3.6 de Ferziger et al., *"Computational Methods for Fluid Dynamics"*.

$$R(\phi^n)=(D-C) \cdot \phi^n$$

Les points aux frontières étant imposés, ils ne sont pas gérés par les méthodes `build_convection_term()` et `build_diffusive_term()`. La méthode `apply_boundary_conditions()` s'occupe de les appliquer sur le résultats de la multiplication ($\phi^{n+1}$)

<div align=center>
    <h3>Système d'équation algébrique sous forme matricielle</h3>
    <img src=./images/image_systemeMatrice.png width=450px>
</div>

Ici le code est simplifié pour montrer la logique de résolution de la boucle de résolution temporelle à l'intérieur de la méthode `solve_explicit()` La méthode `apply_boundary_conditions()` vient appliquer les conditions limites directement sur le champ ``self.phi`` de la classe

```python
for n in range(n_steps):
            phi_flat = self.phi.flatten()   # phi mis sous forme vectorielle
            rhs = (D - C).dot(phi_flat)
            
            self.phi += dt * rhs.reshape((self.nx, self.ny))

            # On impose les conditions aux limites
            self.apply_boundary_conditions()

```

```python
def apply_boundary_conditions(self):
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
```

## Méthode implicite
Le méthode implicite est décrite de la manière suivante: 
$$\phi^{n+1} = \phi^n + \Delta t \cdot R (\phi^{n+1})$$
Comme précédemment,
$$R(\phi^{n+1})=(D-C) \cdot \phi^{n+1}= K \cdot \phi^{n+1}$$
On obtient alors
$$\phi^{n+1} = \phi^n + \Delta t \cdot K\phi^{n+1}$$
En mettant $\phi^{n+1}$ en évidence,
$$\phi^n = \phi^{n+1} [I-K\Delta t]=A\phi^{n+1}$$

Avant d'entrer dans la boucle de résolution, les conditions limites sont appliquées à la matrice **A** à par la méthode `replace_rows_A_matrix()`. On résoud alors $A \phi^{n+1}=\phi^n$ pour trouver $\phi^{n+1}$, $\phi^{n}$ étant connu. Dans la boucle de résolution les conditions limites sont imposées au vecteur $\phi^n$ avant chaque pas à l'aide de la méthode `replace_b()` afin de bien respecter les conditions imposées.

La méthode `solve_implicit()` simplifiée:

```python
def solve_implicit(self, dt, t_end, plot_every=5):
        C = self.build_convection_term()
        D = self.build_diffusive_term()
        
        # Construction de l'opérateur A = I - (dt/rho)*(D - C)
        I = eye(self.n_total, format='lil')
        A = I - dt * (D - C)
        A = A.tolil()

        A_csr = self.replace_rows_A_matrix(A)

        # Boucle temporelle
        n_steps = int(t_end / dt)
        print(f"Début de la simulation : {n_steps} pas de temps.")

        for n in range(n_steps):
            b = self.phi.flatten()  # Solution actuelle mise sous forme de vecteur
            b = self.replace_b(b)   # Imposition des conditions limites

            phi_new = spsolve(A_csr, b)
            
            self.phi.flat[:] = phi_new

        return self.phi
```