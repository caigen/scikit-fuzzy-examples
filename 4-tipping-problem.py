import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

# Problem: from service quality to tip amount
x_service = np.arange(0, 10.001, 1.0)
x_tip = np.arange(0, 20.001, 1.0)

# Membership functions
service_low = fuzz.trimf(x_service, [0, 0, 5])
service_middle = fuzz.trimf(x_service, [0, 5, 10])
service_high = fuzz.trimf(x_service, [5, 10, 10])

tip_low = fuzz.trimf(x_tip, [0, 0, 10])
tip_middle = fuzz.trimf(x_tip, [0, 10, 20])
tip_high = fuzz.trimf(x_tip, [10, 20, 20])

# Input: service score
score = 9.0
low_degree = fuzz.interp_membership(x_service, service_low, score)
middle_degree = fuzz.interp_membership(x_service, service_middle, score)
high_degree = fuzz.interp_membership(x_service, service_high, score)
print(low_degree)
print(middle_degree)
print(high_degree)

# Whole config
fig_scale = 1.5
plt.figure(figsize=(6.4 * fig_scale, 4.8 * fig_scale))
row = 2
col = 2

plt.subplot(row, col, 1)
plt.title("Service Quality")
plt.plot(x_service, service_low, label="low", marker=".")
plt.plot(x_service, service_middle, label="middle", marker=".")
plt.plot(x_service, service_high, label="high", marker=".")
plt.plot(score, 0.0, label="score", marker="D")
plt.plot(score, low_degree, label="low degree", marker="o")
plt.plot(score, middle_degree, label="middle degree", marker="o")
plt.plot(score, high_degree, label="high degree", marker="o")
plt.legend(loc="upper left")

plt.subplot(row, col, 2)
plt.title("Tip")
plt.plot(x_tip, tip_low, label="low", marker=".")
plt.plot(x_tip, tip_middle, label="middle", marker=".")
plt.plot(x_tip, tip_high, label="high", marker=".")
plt.legend(loc="upper left")

# =======================================
# Mamdani (max-min) inference method:
# * min because of logic 'and' connective.
# 1) low_degree <-> tip_low
# 2) middle_degree <-> tip_middle
# 3) high_degree <-> tip_high
activation_low = np.fmin(low_degree, tip_low)
activation_middle = np.fmin(middle_degree, tip_middle)
activation_high = np.fmin(high_degree, tip_high)

plt.subplot(row, col, 3)
plt.title("Tip Activation: Mamdani Inference Method")
plt.plot(x_tip, activation_low, label="low tip", marker=".")
plt.plot(x_tip, activation_middle, label="middle tip", marker=".")
plt.plot(x_tip, activation_high, label="high tip", marker=".")
plt.legend(loc="upper left")

print(activation_low)
print(activation_middle)
print(activation_high)

# Apply the rules:
# * max for aggregation, like or the cases
aggregated = np.fmax(
    activation_low,
    np.fmax(activation_middle, activation_high))

# Defuzzification
tip_centroid = fuzz.defuzz(x_tip, aggregated, 'centroid')
tip_bisector = fuzz.defuzz(x_tip, aggregated, 'bisector')
tip_mom = fuzz.defuzz(x_tip, aggregated, "mom")
tip_som = fuzz.defuzz(x_tip, aggregated, "som")
tip_lom = fuzz.defuzz(x_tip, aggregated, "lom")

plt.subplot(row, col, 4)
plt.title("Aggregation and Defuzzification")
plt.plot(x_tip, aggregated, label="fuzzy result", marker=".")
plt.plot(tip_centroid, 0.0, label="centroid", marker="o")
plt.plot(tip_bisector, 0.0, label="bisector", marker="o")
plt.plot(tip_mom, 0.0, label="mom", marker="o")
plt.plot(tip_som, 0.0, label="som", marker="o")
plt.plot(tip_lom, 0.0, label="lom", marker="o")
plt.legend(loc="upper left")

plt.savefig("img/4-tipping-problem.png")
plt.show()
