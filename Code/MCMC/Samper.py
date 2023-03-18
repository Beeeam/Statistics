from scipy.integrate import quad, dblquad, tplquad
import numpy as np


class Sampler:
    def __init__(self, N, p, x_range, y_range=None, z_range=None, burn=None, seed=None):
        #hyper-parameters
        self.p = p #pdf may not be regularized
        self.x_range = x_range 
        self.y_range = y_range
        self.z_range = z_range
        self.N = N #number of samples
        self.burn = burn        
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
               
        #initialize
        self.scalar = 0.
        self._scalar(self.p, self.x_range, self.y_range, self.z_range)
        # assert self.scalar > 0, 'the pdf has a non-positive integral !!!'
        self.pdf = None
        if self.scalar<0:
            print('please calculate the pdf by hands')
            self.pdf = self.p
        else:
            self.pdf = self._pdf()
        
    
    def MH_Sampler(self, pdf, size=None, sigma=0.5):
        '''
        metropolis_hastings algorithm for 1D sampling
        '''
        if size is None:
            size = self.N
        def proposal_pdf(x, sigma):
            return np.random.normal(x, sigma)
        x = np.zeros(size+1)
        x[0] = np.random.normal(0,1)
        for i in range(1, size+1):
            x_star = proposal_pdf(x[i-1], sigma)
            alpha = min(1, pdf(x_star)/pdf(x[i-1]))
            u = np.random.uniform()
            if u < alpha:
                x[i] = x_star
            else:
                x[i] = x[i-1]
        return x[1:]
    
    def Gibbs_2D(self, n_iter=10000):
        n_iter = n_iter
        assert self.burn+self.N < n_iter, 'burn is larger than n_iter!'
        x = np.zeros(n_iter)
        y = np.zeros(n_iter)
        for i in range(1, n_iter):
            x[i] = self.MH_Sampler(self._con_x(y=y[i-1],1)
            y[i] = self.MH_Sampler(self._con_y(x=x[i],1)
        
        return x[self.burn : self.burn+self.N], y[self.burn : self.burn+self.N]
    
    def Gibbs_3D(self, n_iter=10000):
        n_iter = n_iter
        assert self.burn+self.N < n_iter, 'burn is larger than n_iter!'
        x = np.zeros(n_iter)
        y = np.zeros(n_iter)
        z = np.zeros(n_iter)
        for i in range(1, n_iter):
            x[i] = self.MH_Sampler(self._con_x(y=y[i-1],z=z[i-1]),1)
            y[i] = self.MH_Sampler(self._con_y(x=x[i],z=z[i-1]),1)
            z[i] = self.MH_Sampler(self._con_z(x=x[i],y=y[i]),1)
        
        return x[self.burn : self.burn+self.N], y[self.burn : self.burn+self.N], z[self.burn : self.burn+self.N]
            
    def _scalar(self, p ,x_range, y_range, z_range):
        if y_range is None and z_range is None:
            self.scalar = quad(p, x_range[0], x_range[1])[0]
        elif y_range is not None and z_range is None:
            self.scalar = dblquad(p, x_range[0], x_range[1], lambda x: y_range[0], lambda x: y_range[1])[0]
        elif y_range is not None and z_range is not None:
            self.scalar = tplquad(p, x_range[0], x_range[1], lambda x: y_range[0], lambda x: y_range[1], lambda x, y: z_range[0], lambda x, y:z_range[1], epsabs=1e-10, epsrel=1e-10)[0]
    
    def _pdf(self):
        if self.scalar == 1.:
            return self.p
        else:
            return lambda x,y,z: self.p(x,y,z) / self.scalar
    
    def _prob_x(self, x, dim = 3):
        if dim == 2:
            return quad(lambda y: self.pdf(x,y), self.y_range[0], self.y_range[1])[0]
        elif dim == 3:
            return dblquad(lambda y, z: self.pdf(x,y,z), self.y_range[0], self.y_range[1], self.z_range[0], self.z_range[1])[0] + 1e-10
        else:
            raise NotImplementedError('Error! Only support dimension 2 and 3!')
            
    def _prob_y(self,y, dim = 3):
        if dim == 2:
            return quad(lambda x: self.pdf(x,y), self.x_range[0], self.x_range[1])[0]
        elif dim == 3:
            return dblquad(lambda x, z: self.pdf(x,y,z), self.x_range[0], self.x_range[1], self.z_range[0], self.z_range[1])[0] + 1e-10
        else:
            raise NotImplementedError('Error! Only support dimension 2 and 3!')
    
    def _prob_z(self,z):
        return dblquad(lambda x, y: self.pdf(x,y,z), self.x_range[0], self.y_range[1], self.z_range[0], self.z_range[1])[0] + 1e-10
    
    def _con_x(self, y, z=None):
        if z is None:
            return lambda x: self.pdf(x,y) / (self._prob_x(x, dim = 2) + 1e-10)
        else:
            return lambda x: self.pdf(x,y,z) / (self._prob_x(x, dim = 3) + 1e-10)

    
    def _con_y(self, x, z=None):
        if z is None:
            return lambda y: self.pdf(x,y) / (self._prob_y(y, dim = 2) + 1e-10)
        else:
            return lambda y: self.pdf(x,y,z) / (self._prob_y(y, dim = 3) + 1e-10)


    
    def _con_z(self, x, y):
        # random generate 10 point from [-1,1]
        return lambda z: self.pdf(x,y,z) / (self._prob_z(z) + 1e-10)
