import numpy as np
import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection="3d")

a = -2
k = 0
h = 3
x = np.linspace(0, 1, 100)
z = a * (x - k)**2 + h
y = np.zeros_like(x)

p = np.array([1, 2, 3])
q = p[1:]
print(q, np.concatenate(([1], q)))

ax.plot(x, y, z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()