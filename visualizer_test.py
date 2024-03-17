import matplotlib as mpl  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
import numpy as np
import time
# mpl.use('TkAgg')  # Use an appropriate interactive backend

# ****************************************************************************
# *                               Create data                                *
# ****************************************************************************
np.random.seed(11)
n = 200
x = y = np.linspace(-10, 10, n)
z = np.random.randn(n)*3 + 2


# ****************************************************************************
# *                                 Plot data                                *
# ****************************************************************************
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Show the figure, adjust it with the mouse
plt.show(block=False)
# plt.ion()
# plt.show()

while True:
    plt.pause(1.)
    # time.sleep(1.)
    print('ax.azim {}'.format(ax.azim))
    print('ax.elev {}'.format(ax.elev))

