import numpy as np
from fluid_grid import FluidGrid
import matplotlib.pyplot as plt

class FluidVisualization:

    def __init__(self, grid):
        self.grid = grid

    def get_velocity_field(self):
        u_center = 0.5 * (self.grid.u[:, 0:-1] + self.grid.u[:, 1:])   # (ny, nx)
        v_center = 0.5 * (self.grid.v[0:-1, :] + self.grid.v[1:, :])   # (ny, nx)
        
        return u_center, v_center

    def plot_velocity_field(self, title="Velocity Field", ax=None):
        u_combined, v_combined = self.get_velocity_field() 
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            show_now = True
        else:
            show_now = False
        
        ax.quiver(u_combined, v_combined, scale=10)
        ax.set_aspect('equal')
        ax.set_title(title)
        
        if show_now:
            plt.show()

    def plot_divergence(self, solver, title="Divergencia del Campo de Velocidades", ax=None):
        """
        Plotea la divergencia del campo de velocidades actual como heatmap
        """
        # Calcular divergencia del estado actual
        solver.calculate_divergence()
        divergence = self.grid.divergence.copy()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            show_now = True
        else:
            show_now = False
        
        # Crear heatmap
        im = ax.imshow(divergence, cmap='RdBu', origin='lower')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        # Mostrar estadísticas
        div_max = np.max(np.abs(divergence))
        ax.text(0.02, 0.98, f'Div máx: {div_max:.6f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if show_now:
            plt.show()
            
        return divergence

