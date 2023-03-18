import numpy as np
import Sampler
burn = 8000
seed = 7
def p(x,y,z):    
    return  ((1+z**2) * np.exp(-((x-y-1)**2 + y**2)/2))/(2*np.pi*8/3)                             
x_range = [-np.inf, np.inf]
y_range = [-np.inf, np.inf]
z_range = [-1, 1]
N = 1000
sampler = Sampler(N, p, x_range, y_range, z_range, burn, seed=9)
x , y , z = sampler.Gibbs_3D()
E = np.mean((np.abs(x) ** z + y ** 2))
