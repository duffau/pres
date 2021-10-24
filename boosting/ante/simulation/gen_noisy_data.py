import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

n = 10000
k = 10
X = st.norm().rvs(size=(n, k))
threshold = st.chi2.ppf(0.5, k)
noise_std = threshold*0.75
noise = st.norm(loc=0, scale=noise_std).rvs(size=n)

print(f"threshold: {threshold}")
y = ((X*X).sum(axis=1) + noise < threshold)*2 - 1
print(f"y.shape: {y.shape}")
print(f"X.shape: {X.shape}")

print("Saving X and y data...")
np.save("X_noisy.npy", X)
np.save("y_noisy.npy", y)

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
plt.savefig("gen_noisy_data.svg")
plt.close()