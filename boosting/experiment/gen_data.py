import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

n = 10000

x1 = st.uniform(loc=-1, scale=2).rvs(size=n)
x2 = st.uniform(loc=-1, scale=2).rvs(size=n)
label = (x1**2 + x2**2 + x1*x2 < .25)*1
print(label)

colors = np.array(["tab:blue", "tab:red"])
print(colors[label])
fig1, ax = plt.subplots()
ax.set_aspect('equal')
ax.scatter(x1, x2, s=3, c=colors[label])
ax.add_patch(plt.Circle((0, 0), 0.25**0.5, color='black', fill=False))
plt.show()