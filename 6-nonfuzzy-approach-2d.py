from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# for flake8 check
Axes3D

# Whole config
row = 2
col = 2
fig_scale_x = 1.5
fig_scale_y = 1.5

# =============================
fig = plt.figure(figsize=(6.4 * fig_scale_x, 4.8 * fig_scale_y))

ax = fig.add_subplot(row, col, 1, projection="3d")
plt.title("equal ratio")

food = np.arange(0, 10.01, 0.5)
service = np.arange(0, 10.01, 0.5)
f, s = np.meshgrid(food, service, indexing="ij")

tip = (0.20 / 20) * (s + f) + 0.05

ax.set_xlabel("Service")
ax.set_xlim(0, 10)
ax.set_ylabel("Food")
ax.set_xlim(0, 10)
ax.set_zlabel("Tip")
ax.set_zlim(0.05, 0.25)
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
surf = ax.plot_surface(s, f, tip, cmap=cm.coolwarm, linewidth=0)

# ==============================
ax2 = fig.add_subplot(row, col, 2, projection="3d")
plt.title("service accounts for 80%")

service_ratio = 0.8
tip = service_ratio * (0.20 / 10 * s + 0.05) \
    + (1 - service_ratio) * (0.20 / 10 * f + 0.05)

ax2.set_xlabel("Service")
ax2.set_xlim(0, 10)
ax2.set_ylabel("Food")
ax2.set_xlim(0, 10)
ax2.set_zlabel("Tip")
ax2.set_zlim(0.05, 0.25)
ax2.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
surf2 = ax2.plot_surface(s, f, tip, cmap=cm.coolwarm, linewidth=0)

# ===============================
ax3 = fig.add_subplot(row, col, 3, projection="3d")
plt.title("complicated ratio by hard code")

print(len(f))
print(len(s))
for i in range(0, len(f)):
    for j in range(0, len(s)):
        if s[i, j] < 3:
            tip[i, j] = ((0.10 / 3) * s[i, j] + 0.05) * service_ratio \
                + (1 - service_ratio) * (0.20 / 10 * f[i, j] + 0.05)
        elif s[i, j] < 7:
            tip[i, j] = 0.15 * service_ratio \
                + (1 - service_ratio) * (0.20 / 10 * f[i, j] + 0.05)
        else:
            tip[i, j] = ((0.10 / 3) * (s[i, j] - 7) + 0.15) * service_ratio \
                + (1 - service_ratio) * (0.20 / 10 * f[i, j] + 0.05)

ax3.set_xlabel("Service")
ax3.set_xlim(0, 10)
ax3.set_ylabel("Food")
ax3.set_xlim(0, 10)
ax3.set_zlabel("Tip")
ax3.set_zlim(0.05, 0.25)
ax3.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
surf3 = ax3.plot_surface(s, f, tip, cmap=cm.coolwarm, linewidth=0)

# ===============================
ax4 = fig.add_subplot(row, col, 4, projection="3d")
plt.title("complicated ratio by hard code")

ax4.set_xlabel("Food")
ax4.set_xlim(0, 10)
ax4.set_ylabel("Service")
ax4.set_xlim(0, 10)
ax4.set_zlabel("Tip")
ax4.set_zlim(0.05, 0.25)
ax4.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
surf4 = ax4.plot_surface(f, s, tip, cmap=cm.coolwarm, linewidth=0)

plt.savefig("img/6-nonfuzzy-approach-2d.png")
plt.show()
