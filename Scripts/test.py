import numpy as np


a = np.arange(0,101,1)
# print(a)

b = np.linspace(0,100,101)
# print(b)

# start = 100
# end = 350

# sfreq = 512
# nb_pts = (end - start) * 512 / 1000

# test = np.linspace(start,end,int(nb_pts))
# print(len(test))



print(b)
b1 = b[(b > 50) & (b%2 == 0)]
print(b1)


import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(100, 200, 1000)  # ton axe

seuil_max = 0.5
mu = 50      # moyenne
sigma = 30   # écart-type

g = seuil_max * np.exp(- (x - mu)**2 / (2 * sigma**2))


print(g)
plt.plot(x,g)
plt.show()
