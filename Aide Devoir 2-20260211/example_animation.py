# Animation of temperature field over time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
Lx, Ly = 1, 1
Nx, Ny = 80, 80
tf = 1.0 # total simulation time
dt = 0.0045 # time-step

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
Nit = int(tf // dt) + 1

t = np.linspace(0, tf, Nit)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
XX, YY, = np.meshgrid(x, y, indexing='ij')

XX = XX[:,:,np.newaxis]
YY = YY[:,:,np.newaxis]
t_broadcasted = t[np.newaxis, np.newaxis,:]

# Example of Temperature evolution with a period of T = tf/4
period = tf/4
omega = 2*np.pi/period
temperature = np.cos(XX)*np.cos(YY)*np.cos(omega*t_broadcasted)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)  
ax.set_xlabel('x')  
ax.set_ylabel('y')  

T = np.zeros_like(temperature[:,:,0])
X1 = XX[:,:,0]
Y1 = YY[:,:,0]

global contf, cont
cmap = 'bwr'
contour_values = np.linspace(-1, 1, 21)
contf = ax.contourf(X1, Y1, T, levels=contour_values, cmap=cmap)
cont = ax.contour(X1, Y1, T, levels=contour_values, colors='k')
cbar = fig.colorbar(contf, ticks=np.linspace(-1, 1, 11))
cbar.set_label('$\phi$')

# Add time display box
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                    fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def animate(i):
    """Updates the temperature field for frame i."""
    global contf, cont, time_text

    current_time = t[i]
    time_text.set_text(f'Time: {current_time:.4f} s')    

    Z = temperature[:,:,i]

    for collection in contf.collections:
        collection.remove()
    for collection in cont.collections:
        collection.remove()

    # Update the quiver object's U and V data
    contf = ax.contourf(X1, Y1, Z, levels=contour_values, cmap=cmap)
    cont = ax.contour(X1, Y1, Z, levels=contour_values, colors='k')

    return contf.collections + cont.collections + [time_text]

frames = Nit
interval = 100
video_time = 5.0
fps = int((frames)/video_time)
print(f'{fps=}')
ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, 
                              blit=False, repeat=False)
ani.save('Temperature_evolution_example.gif', fps=fps)