import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

# [0, 10]
start = 0
stop = 10 + 0.001
step = 0.25
x = np.arange(start, stop, step)
print(x)

# Triangular membership function
trimf = fuzz.trimf(x, [0, 5, 10])

# Trapezoidal membership function
trapmf = fuzz.trapmf(x, [0, 2, 8, 10])

# Sigmoid membership function
center = 5.0
width_control = 2.0
sigmf = fuzz.sigmf(x, center, width_control)

# S-function
foot = 1.0
ceiling = 9.0
smf = fuzz.smf(x, foot, ceiling)

# Z-function
zmf = fuzz.zmf(x, foot, ceiling)

# Pi-function
pimf = fuzz.pimf(x, 0.0, 4.0, 5.0, 10.0)

# Gaussian function
mean = 5.0
sigma = 1.25
gaussmf = fuzz.gaussmf(x, mean, sigma)

# Generalized Bell-Shaped function
width = 2.0
slope = 4.0
center = 5.0
gbellmf = fuzz.gbellmf(x, width, slope, center)

fig_scale = 1.5
plt.figure(figsize=(6.4 * fig_scale, 4.8 * fig_scale))

# 3 rows, 1 col, index from 1
plt.subplot(311)
plt.title("Fuzzy Membership Function")
plt.plot(x, trimf, label="Triangle")
plt.plot(x, trapmf, label="Trapezoidal")
plt.legend(loc="upper right")

plt.subplot(312)
plt.plot(x, sigmf, label="Sigmoid")
plt.plot(x, smf, label="S-function")
plt.plot(x, zmf, label="Z-function")
plt.plot(x, pimf, label="Pi-function")
plt.legend(loc="upper right")

plt.subplot(313)
plt.plot(x, gaussmf, label="Gaussan")
plt.plot(x, gbellmf, label="Bell-Shaped")
plt.legend(loc="upper right")

plt.savefig("img/1-memship-function.png")
plt.show()
