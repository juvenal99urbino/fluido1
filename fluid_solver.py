import numpy as np

class FluidSolver:
    def __init__(self, grid, over_relaxation=1.0):
        self.grid = grid
        self.over_relaxation = over_relaxation

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
        # acolchonamos para poder indexar vecinos sin ifs
        ctp = np.pad(ct, pad_width=1, mode='constant', constant_values=0)

        # vecinos relativos en acolchado:
        # centro = ctp[1:-1, 1:-1]
        cC = ctp[1:-1, 1:-1]
        cL = ctp[1:-1, 0:-2]
        cR = ctp[1:-1, 2:  ]
        cB = ctp[0:-2, 1:-1]
        cT = ctp[2:  , 1:-1]

        # pesos por cara: cara existe si ambas celdas son fluidas
        wL = cC * cL
        wR = cC * cR
        wB = cC * cB
        wT = cC * cT

        # suma de pesos (denominador por celda)
        self.grid.nsum = wL + wR + wB + wT  # (ny, nx)

        # guardamos también para usarlos en seidel_step sin recalcular
        self._wL, self._wR, self._wB, self._wT = wL, wR, wB, wT

    def seidel_step(self):
        """
        Distribuye la divergencia de cada celda a sus cuatro caras,
        ponderando por las paredes vía cell_type.
        """
        #self.calculate_divergence()
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
                    - self.grid.u[i, j] + self.grid.u[i, j+1]
                    - self.grid.v[i, j] + self.grid.v[i+1, j]
                )*self.over_relaxation

                # repartir en proporción a cada cara existente
                # (si una cara está bloqueada, su peso es 0 y no recibe nada)
                # signos: +L, -R, +B, -T (coherentes con la definición de d)
                u[i, j]   += d * (wL[i, j] / s_ij)  # izquierda
                u[i, j+1] -= d * (wR[i, j] / s_ij)  # derecha
                v[i, j]   += d * (wB[i, j] / s_ij)  # abajo
                v[i+1, j] -= d * (wT[i, j] / s_ij)  # arriba

    def solve(self, iterations=100):
        for _ in range(iterations):
            self.seidel_step()
