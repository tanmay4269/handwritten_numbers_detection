from utils import *

w = np.random.uniform(low=-1, high=1, size=(16,16))

a = np.ones((16, 1))

b = np.ones((16, 1))

z = np.matmul(w, a) + b

print(z)