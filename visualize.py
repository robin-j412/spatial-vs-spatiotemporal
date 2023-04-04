import numpy as np
from data import EXPE, patch
import matplotlib.pyplot as plt

X = np.load(EXPE + '/raw_data/sits.npy')

X_vis = np.moveaxis(X[0, :, :, [2, 1, 0]]*2, 0, -1)

plt.imshow(X_vis)
plt.show()

dx, dy = 128, 128

# Custom (rgb) grid color
grid_color = [255,255,255]

# Modify the image to include the grid
X_vis[:,::dy,:] = grid_color
X_vis[::dx,:,:] = grid_color

# Show the result
plt.imshow(X_vis)
plt.show()

patches = patch(X, 32)

fig, axes = plt.subplots(3, 21, figsize=(9, 2))

for i in range(3):
    for j in range(21):
        img = np.moveaxis(patches[i, j, :, :, [2, 1, 0]]*3, 0, -1)
        axes[i, j].imshow(img)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])


plt.show()