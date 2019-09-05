import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

# [0, 10]
x = np.arange(11)
mfx = fuzz.trimf(x, [0, 5, 10])

print(x)
print(mfx)

plt.title("Fuzzy Membership Function")
plt.plot(x, mfx)

plt.savefig("img/0-getting-started.png")
plt.show()
