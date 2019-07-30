import matplotlib.pylab as plt
import numpy as np

handle = open("D:\CAt.txt", "r")
data = handle.read()
a = np.array(data)
print(data)

handle.close()
plt.plot(a)
plt.xlabel('x')

plt.show()
