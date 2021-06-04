import numpy as np
import pickle
import sys

# Read in pickle matrix
filename = sys.argv[1]
p = pickle.load(open(filename, "br"))

# weight & bias
w, b = p
w = w.reshape(w.shape[:2])

# Plot
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(w, cmap='gray')
plt.colorbar()
plt.title("Rows output channels, columns input channels")
plt.figure()
plt.plot(b)
plt.show()