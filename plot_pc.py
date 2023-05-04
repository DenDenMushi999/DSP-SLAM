import numpy as np
import matplotlib.pyplot as plt

pts = np.load('pt_cloud.npy')
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(pts[:,0], pts[:,1], pts[:,2])
plt.show()

