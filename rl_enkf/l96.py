

class L96:
    def __init__(self, N, F):
        self.N = N
        self.F = F

    def dx(self, x0, t):
        # dx_i/dt = (x_{i + 1} - x_{i - 2}) * x_{i - 1} - x_i + F
        dxs = np.zeros(self.N)
        dxs[0] = (x0[1] - x0[self.N - 2]) * x0[self.N - 1] - x0[0] + self.F
        dxs[1] = (x0[2] - x0[self.N - 1]) * x0[0] - x0[1] + self.F
        dxs[self.N - 1] = (x0[0] - dxs[self.N - 3]) * x0[self.N - 2] - x0[self.N - 1] + self.F
        for i in range(2, self.N - 1):
            dxs[i] = (x0[i+1] - x0[i - 2]) * x0[i - 1] - x0[i] + F

        return dxs
