import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Same data as Stage 2
fs = 100
t = np.arange(0, 20, 1/fs)

x = 0.6*np.sin(2*np.pi*5*t)
y = 0.4*np.sin(2*np.pi*12*t)
z = 0.05*np.random.randn(len(t)) + 1  # gravity

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(i):
    ax.cla()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])

    ax.quiver(0,0,0, x[i],0,0, color='r', label='X')
    ax.quiver(0,0,0, 0,y[i],0, color='g', label='Y')
    ax.quiver(0,0,0, 0,0,z[i], color='b', label='Z')

    ax.set_title("Accelerometer Working Model (X, Y, Z)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

ani = FuncAnimation(fig, update, frames=len(t), interval=50)
plt.show()
