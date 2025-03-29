import numpy as np
import matplotlib.pyplot as plt
from Classification import *
from Read_data import *

X,y = read_samples()

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()