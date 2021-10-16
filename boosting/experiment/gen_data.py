import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

n = 10000
k = 10
X = st.norm().rvs(size=(n, k))
threshold = st.chi2.ppf(0.5, k)
print(f"threshold: {threshold}")
y = ((X*X).sum(axis=1) < threshold)*2 - 1
print(f"y.shape: {y.shape}")
print(f"X.shape: {X.shape}")

print("Saving X and y data...")
np.save("X.npy", X)
np.save("y.npy", y)

x1, x2 , *ignore = X.T
colors = np.array(["tab:blue", "tab:red"])
fig1, ax = plt.subplots()
ax.set_aspect('equal')
y_color_indic = (y+1)//2
ax.scatter(x1, x2, s=3, c=colors[y_color_indic])
plt.title("2D slice of 10D ball")
plt.xlabel("x1")
plt.ylabel("x2")
# ax.add_patch(plt.Circle((0, 0), threshold**0.5, color='black', fill=False))
plt.savefig("gen_data.svg")
plt.close()