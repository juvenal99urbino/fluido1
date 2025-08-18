from fluid_grid import FluidGrid
from fluid_solver import FluidSolver
from fluid_visualization import FluidVisualization

import matplotlib.pyplot as plt
import numpy as np

def test_divergence(grid, viz, solver):
    grid.customtest()  # Reinicializar para el test
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Figura 1: Divergencia inicial
    viz.plot_divergence(solver, title="Divergencia Inicial", ax=ax1)
    solver.solve(iterations=30)  # Ejecutar el solver
    # Figura 2: Divergencia final
    viz.plot_divergence(solver, title="Divergencia Final", ax=ax2)

    plt.tight_layout()
    plt.show()

def test_quiver(grid, viz, solver):
    """
    Test para visualizar el campo de velocidades antes y después del solver
    """
    grid.customtest()

    # Figura 1: Campo de velocidades
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    viz.plot_velocity_field(title="Velocidades Iniciales", ax=ax1)

    solver.solve(iterations=20)

    viz.plot_velocity_field(title="Velocidades (sin divergencia)", ax=ax2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid = FluidGrid(nx=50, ny=50, dx=1.0)
    viz = FluidVisualization(grid)
    solver = FluidSolver(grid, over_relaxation=1.0)
    
    # Test 1: Campo de velocidades (quiver)
    #test_quiver(grid, viz, solver)
    
    # Test 2: Divergencia (heatmaps)
    test_divergence(grid, viz, solver)




