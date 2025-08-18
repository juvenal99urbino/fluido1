import numpy as np

class FluidGrid:
    def __init__(self, nx, ny, dx):
        self.nx = nx
        self.ny = ny
        self.dx = dx

        self.u = np.zeros((ny, nx+1))  # x-component of velocity
        self.v = np.zeros((ny+1, nx))  # y-component of velocity
        self.temp_u = np.zeros((ny, nx+1))
        self.temp_v = np.zeros((ny+1, nx))

        self.pressure = np.zeros((ny, nx))
        self.smoke= np.zeros((ny, nx))
        self.temp_smoke = np.zeros((ny, nx))
        self.divergence = np.zeros((ny, nx))
        
        #Marca que casillas son accesibles (1) o no (0)
        self.cell_type = np.ones((ny, nx))
        self.nsum = np.zeros((ny, nx))

        # Hacer bordes no accesibles (0)
        self.cell_type[0, :] = 0    # Borde izquierdo
        self.cell_type[-1, :] = 0   # Borde derecho
        self.cell_type[:, 0] = 0    # Borde inferior
        self.cell_type[:, -1] = 0   # Borde superior

    def randomize(self):
        self.u = np.random.randn(self.ny, self.nx+1)
        self.v = np.random.randn(self.ny+1, self.nx)

    def customtest(self):
        self.u = np.zeros((self.ny, self.nx+1))  # x-component of velocity
        self.v = np.zeros((self.ny+1, self.nx))  # y-component of velocity
        self.u[20:30, 20:30]=1

    def add_source(self):
        self.u[33:38,10:20]=1
        self.smoke[33:38,10:20]=1


