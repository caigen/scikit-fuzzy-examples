import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

# [0, 10]
delta = 0.001 
start = 0
stop = 10 + delta
step = 0.5
x = np.arange(start, stop + delta, step)

# Triangular membership function
x1 = np.arange(0, 5 + delta, step)
trimf = fuzz.trimf(x1, [0, 2.5, 5])

# Trapezoidal membership function
x2 = np.arange(4, 10 + delta, step)
trapmf = fuzz.trapmf(x2, [4, 6, 8, 10])

# fuzzy logic
tri_not = fuzz.fuzzy_not(trimf)
trap_not = fuzz.fuzzy_not(trapmf)
x3, tri_trap_and = fuzz.fuzzy_and(x1, trimf, x2, trapmf)
x3, tri_trap_or = fuzz.fuzzy_or(x1, trimf, x2, trapmf)
print(x2)
print(x3)

# Whole config
fig_scale = 1.5
plt.figure(figsize = (6.4 * fig_scale, 4.8 * fig_scale))
row = 3
col = 1
#plt.suptitle("Fuzzy Logic")

plt.subplot(row, col, 1)
plt.title("Fuzzy Logic")
#plt.xticks(x)
plt.plot(x1, trimf, label = "Triangle", marker = ".")
plt.plot(x2, trapmf, label = "Trapezoidal", marker = ".")
plt.legend(loc = "upper right")

plt.subplot(row, col, 2)
#plt.xticks(x)
plt.plot(x1, tri_not, label = "Fuzzy NOT Triangle", marker = "x")
plt.plot(x2, trap_not, label = "Fuzzy NOT Trapezoidal", marker = "x")
plt.legend(loc = "upper right")

plt.subplot(row, col, 3)
#plt.xticks(x)
plt.plot(x3, tri_trap_and, label = "Fuzzy AND", marker = "^")
plt.plot(x3, tri_trap_or, label = "Fuzzy OR", marker = "v")
plt.legend(loc = "upper right")

plt.savefig("img/2-fuzzy-logic.png")
plt.show()

