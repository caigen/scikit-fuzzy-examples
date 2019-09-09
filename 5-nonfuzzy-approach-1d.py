import numpy as np
from matplotlib import pyplot as plt

# Whole config
fig_scale_x = 1.8
fig_scale_y = 1.0
plt.figure(figsize=(6.4 * fig_scale_x, 4.8 * fig_scale_y))
row = 1
col = 2

# [0, 10]
service = np.arange(0, 10.01, 0.5)
tip = 0.15 * np.ones(np.size(service))

plt.subplot(row, col, 1)
plt.title("const value")
plt.plot(service, tip)
plt.xlabel("Service")
plt.ylabel("Tip")
plt.xlim(0, 10)
plt.ylim(0.05, 0.25)

tip = (0.20 / 10) * service + 0.05
plt.subplot(row, col, 2)
plt.title("linear mapping")
plt.plot(service, tip)
plt.xlabel("Service")
plt.ylabel("Tip")
plt.xlim(0, 10)
plt.ylim(0.05, 0.25)

plt.savefig("img/5-nonfuzzy-approach-1d.png")
plt.show()
