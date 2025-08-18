import numpy as np

class FluidSolver:
    def __init__(self, grid, over_relaxation=1.0, dt=0.1):
        self.grid = grid
        self.over_relaxation = over_relaxation
        self.dt = dt

    def calculate_divergence(self):
        # divergencia en centros de celda (ny, nx)
        for j in range(self.grid.nx):
            for i in range(self.grid.ny):
                self.grid.divergence[i, j] = (
                    - self.grid.u[i, j] + self.grid.u[i, j+1]
                    - self.grid.v[i, j] + self.grid.v[i+1, j]
                )

    def calculate_nsum(self):
        """
        s = wL + wR + wB + wT
        donde cada w* = cell(i,j) * cell(vecino)
        """
        ct = self.grid.cell_type  # (ny, nx)
        ctp = np.pad(ct, pad_width=1, mode='constant', constant_values=0)

        cC = ctp[1:-1, 1:-1]
        cL = ctp[1:-1, 0:-2]
        cR = ctp[1:-1, 2:  ]
        cB = ctp[0:-2, 1:-1]
        cT = ctp[2:  , 1:-1]

        wL = cC * cL
        wR = cC * cR
        wB = cC * cB
        wT = cC * cT

        self.grid.nsum = wL + wR + wB + wT  # (ny, nx)
        self._wL, self._wR, self._wB, self._wT = wL, wR, wB, wT

    def seidel_step(self):
        """
        Distribuye la divergencia de cada celda a sus cuatro caras,
        ponderando por las paredes vía cell_type.
        """
        # self.calculate_divergence()  # no es necesario si usamos residuo local
        self.calculate_nsum()

        ny, nx = self.grid.ny, self.grid.nx
        u, v = self.grid.u, self.grid.v
        s = self.grid.nsum
        wL, wR, wB, wT = self._wL, self._wR, self._wB, self._wT

        for j in range(nx):
            for i in range(ny):
                s_ij = s[i, j]
                if s_ij <= 0:
                    continue  # celda bloque o completamente rodeada de paredes

                d = (
                    - u[i, j] + u[i, j+1]
                    - v[i, j] + v[i+1, j]
                ) * self.over_relaxation

                # signos: +L, -R, +B, -T
                u[i, j]   += d * (wL[i, j] / s_ij)
                u[i, j+1] -= d * (wR[i, j] / s_ij)
                v[i, j]   += d * (wB[i, j] / s_ij)
                v[i+1, j] -= d * (wT[i, j] / s_ij)

    def solve(self, iterations=300):
        for _ in range(iterations):
            self.seidel_step()

    def sample_field(self, field, x, y):
        "Muestrea el campo field en las coordenadas (x, y)"
        dx = self.grid.dx
        ny, nx = field.shape

        x = np.asarray(x)
        y = np.asarray(y)

        # índices enteros de celda "inferior-izquierda"
        x0 = np.floor(x / dx)
        y0 = np.floor(y / dx)

        # clamp para que x0+1 <= nx-1 y y0+1 <= ny-1
        x0 = np.clip(x0, 0, nx - 2).astype(np.int64)
        y0 = np.clip(y0, 0, ny - 2).astype(np.int64)

        # fracciones dentro de la celda
        sx = (x - x0 * dx) / dx
        sy = (y - y0 * dx) / dx

        # pesos
        w00 = (1.0 - sx) * (1.0 - sy)
        w10 = sx * (1.0 - sy)
        w01 = (1.0 - sx) * sy
        w11 = sx * sy

        # índices vecinos
        x1 = x0 + 1
        y1 = y0 + 1

        # bilinear
        v = (field[y0, x0] * w00 +
             field[y0, x1] * w10 +
             field[y1, x0] * w01 +
             field[y1, x1] * w11)

        return v

    def advect_u(self):
        # preservar bordes y valores no actualizados
        self.grid.temp_u[:] = self.grid.u.copy()

        # i: todas las filas de u (0..ny-1), j: caras interiores en x (1..nx-1)
        for i in range(0, self.grid.ny):
            for j in range(1, self.grid.nx):
                x0 = self.grid.dx * j
                y0 = self.grid.dx * (i + 0.5)

                u0 = self.grid.u[i, j]
                # promedio de v en las 4 esquinas alrededor del punto u
                v0 = (self.grid.v[i, j-1] + self.grid.v[i+1, j-1] +
                      self.grid.v[i, j]   + self.grid.v[i+1, j]) * 0.25

                # FIX-1: no restar offsets aquí; solo backtrace físico
                x1 = x0 - self.dt * u0          # ← FIX-1
                y1 = y0 - self.dt * v0          # ← FIX-1

                # FIX-2: compensar offset MAC de u en el muestreo (y - 0.5*dx)
                self.grid.temp_u[i, j] = self.sample_field(
                    self.grid.u, x1, y1 - 0.5 * self.grid.dx
                )                                # ← FIX-2

        # copiar valores actualizados de vuelta (preserva bordes)
        self.grid.u = self.grid.temp_u.copy()     # ← FIX-3

    def advect_v(self):
        # preservar bordes y valores no actualizados
        self.grid.temp_v[:] = self.grid.v.copy()

        # i: caras interiores en y (1..ny-1), j: todas las columnas 0..nx-1
        for i in range(1, self.grid.ny):
            for j in range(0, self.grid.nx):
                x0 = self.grid.dx * (j + 0.5)
                y0 = self.grid.dx * i

                u0 = (self.grid.u[i-1, j] + self.grid.u[i-1, j+1] +
                      self.grid.u[i,   j] + self.grid.u[i,   j+1]) * 0.25
                v0 = self.grid.v[i, j]

                x1 = x0 - self.dt * u0
                y1 = y0 - self.dt * v0

                # FIX-4: muestrear v con su offset MAC (x - 0.5*dx)
                self.grid.temp_v[i, j] = self.sample_field(
                    self.grid.v, x1 - 0.5 * self.grid.dx, y1
                )                                # ← FIX-4

        # copiar valores actualizados de vuelta (preserva bordes)
        self.grid.v = self.grid.temp_v.copy()     # ← FIX-5

    def add_gravity(self, g=9.81):
        """Añade un término de gravedad al campo de velocidades."""
        self.grid.v[:, :] -= g * self.dt

    def general_step(self):
        """Un paso general: advección, gravedad y corrección de presión."""
        #self.add_gravity()
        self.solve()
        self.advect_u()
        self.advect_v()


