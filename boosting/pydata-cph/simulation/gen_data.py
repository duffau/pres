import numpy as np
import matplotlib.pyplot as plt
from utils import gen_spherical_data

N = 10000
K = 10
SEED = 123

X, y = gen_spherical_data(
    n=N, 
    k=K,
    seed=SEED
)

print("Saving X and y data...")
np.save("data/X.npy", X)
np.save("data/y.npy", y)

# Plotting on reduced example data
N = 5000
K = 2
SEED = 123

X, y = gen_spherical_data(
    n=N, 
    k=K,
    seed=SEED
)

x1, x2 , *ignore = X.T
colors = np.array(["tab:blue", "tab:red"])
fig1, ax = plt.subplots()
fig1.set_size_inches(5.5, 5.5)
ax.set_aspect('equal')
y_color_indic = (y+1)//2
ax.scatter(x1, x2, s=3, c=colors[y_color_indic])
plt.title("2D example of nested spheres")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.savefig("plots/gen_data.png")
plt.close()
