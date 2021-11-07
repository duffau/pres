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

# Plotting on reduced data
n_max = 8000
x1, x2 , *ignore = X.T
x1, x2 = x1[0:n_max], x2[0:n_max]
y = y[0:n_max]
colors = np.array(["tab:blue", "tab:red"])
fig1, ax = plt.subplots()
ax.set_aspect('equal')
y_color_indic = (y+1)//2
ax.scatter(x1, x2, s=3, c=colors[y_color_indic])
plt.title(f"2D slice of {K}D ball")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("plots/gen_data.png")
plt.tight_layout()
plt.close()
