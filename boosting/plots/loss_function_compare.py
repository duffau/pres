import numpy as np
import matplotlib.pyplot as plt


def exponential_loss(yf):
    return np.exp(-yf) 


def binomial_deviance(yf):
    return np.log(1 + np.exp(-2*yf)) + (1 - np.log(2))


yf = np.arange(-2, 2, step=0.01)

plt.plot(yf, exponential_loss(yf), color="tab:blue", label="exponential loss: $e^{-yf(x)}$")
plt.plot(yf, binomial_deviance(yf), color="tab:red", label="cross-entropy deviance: $\log(1 + e^{-2yf(x)})$")
plt.ylabel("Loss")
plt.xlabel("$yf$")
plt.legend()
plt.tight_layout()
plt.savefig("loss_functions.svg")
plt.close()