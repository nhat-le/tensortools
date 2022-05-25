import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
import tensortools.utils

data = np.random.randn(100, 2)
x, y = data.T

print(data.shape)
prm = tt.utils.tsp_linearize(data)

plt.plot(x, y, '.b')
plt.plot(x[prm], y[prm], '-r')
plt.show()
