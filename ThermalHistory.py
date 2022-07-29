import numpy as np
from   matplotlib import pyplot as plt
from scipy.optimize import fsolve
from   scipy.interpolate import CubicSpline
import warnings
import scipy.special as ss
from  recfast4py import recfast

from   Global import const,constCos
from  BackGroundCosmology import BackGroundCosmology
BG = BackGroundCosmology()

class ThermalHistory:
  """
  This is a class for solving the recombination (and reionization) history of the Universe.
  It holds recombination parameters and functions relevant for the recombination history.
  
  Input Parameters: 
    cosmo (BackgroundCosmology) : The cosmology we use to solve for the recombination history
    Yp                   (float): Primordial helium fraction
    reionization         (bool) : Include reionization or not
    z_reion              (float): Reionization redshift
    delta_z_reion        (float): Reionization width
    helium_reionization  (bool) : Include helium+ reionization
    z_helium_reion       (float): Reionization redshift for helium+
    delta_z_helium_reion (float): Reionization width for helium+

  Attributes:    
    tau_reion            (float): The optical depth at reionization
    z_star               (float): The redshift for the LSS (defined as peak of visibility function or tau=1)

  Functions:
    tau_of_x             (float->float) : Optical depth as function of x=log(a) 
    dtaudx_of_x          (float->float) : First x-derivative of optical depth as function of x=log(a) 
    ddtauddx_of_x        (float->float) : Second x-derivative of optical depth as function of x=log(a) 
    g_tilde_of_x         (float->float) : Visibility function dexp(-tau)dx as function of x=log(a) 
    dgdx_tilde_of_x      (float->float) : First x-derivative of visibility function as function of x=log(a) 
    ddgddx_tilde_of_x    (float->float) : Second x-derivative of visibility function as function of x=log(a)
    Xe_of_x              (float->float) : Free electron fraction dXedx as function of x=log(a) 
    ne_of_x              (float->float) : Electron number density as function of x=log(a) 
  """

  
    
  def __init__(self, BackGroundCosmology,Yp = 0.24, 
      reionization = True, z_reion = 11.0, delta_z_reion = 0.5, 
      helium_reionization = True, z_helium_reion = 3.5, delta_z_helium_reion = 0.5):
    self.cosmo            = BackGroundCosmology
    
    self.Yp               = Yp
    
    self.reionization     = reionization
    self.z_reion          = z_reion
    self.delta_z_reion    = delta_z_reion
    
    self.helium_reionization  = helium_reionization
    self.z_helium_reion       = z_helium_reion
    self.delta_z_helium_reion = delta_z_helium_reion
    
    # derived quanties in terms of efolds
    self.x_reion          = - np.log(self.z_reion + 1.0)
    self.delta_x_reion    = - np.log(self.delta_z_reion + 1.0)
    self.x_helium_reion   = - np.log(self.z_helium_reion + 1.0)
    self.delta_x_helium_reion = - np.log(self.delta_z_helium_reion + 1.0)
    
    
    
    
    
  
  def baryon_to_photon_ratio(self):
      """
      Baryons to phtons ratio defined as  \frac{\pi^{4} k_b T_{cmb} \Omega_{B_0} } { 30 \zeta{3} m_b c^{2} \Omega_{\gamma_{0}}}

      Returns  float
      -------
      
      """
      ival1 = np.pi **(4.0) * const.k_b * BG.TCMB * BG.OmegaB0
      ival2 = 30.0 * ss.zeta(3.0) * const.m_p * const.c **(2.0) * BG.OmegaG0
      ival = ival1 / ival2 
      return ival
  
  def number_density_baryons(self,x):
      """
      Calculate the number density for baryons which is given as
      n_b = \frac{2 \zeta(3)} {\pi^{2} \hbar^{3} c^{3}} (k_b T)^{3}

      Returns float
      -------
      

      """
      ival1 = (2.0 * ss.zeta(3.0)) / (np.pi **(2.0) * const.hbar **(3.0) * const.c **(3.0) ) 
      ival2 = (const.k_b * BG.Temp_to_x(x)) **(3.0)
      return self.baryon_to_photon_ratio() * ival1 * ival2
  
  def number_density_Hydrogen(self,x):
      """
      Calculate the number density for Hydrogen which is given as
      n_H = ( 1 - Y_p) number_density_baryons

      Parameters
      ----------
      x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.
      Returns
      -------
      number_density_Hydrogen : ndarray or float
      The Number denisty of Hydrogen at each e-fold.
      Returns float if input scalar.

      """
      return (1.0 - self.Yp) * self.number_density_baryons(x)
  
    
  def Xe(self,x):
      """
      Calculate the free electron fraction 
      

      Parameters
      ----------
      x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.
      Returns
      -------
      Xe : ndarray or float
      The free electron fraction  at each e-fold.
      Returns float if input scalar.
      
      """
      fHe = self.Yp / (4.0 * (1.0 - self.Yp))      
      F = 1.14
      fDM = 0.0
      zarr, Xe_H, Xe_He, Xe ,TM = recfast.Xe_frac(self.Yp,BG.TCMB,BG.OmegaCDM0, BG.OmegaB0, BG.OmegaDE0,BG.OmegaK0, BG.h, BG.Neff, F, fDM,switch=0,npz = 10**(6),zstart = 10**(7),zend = 0.01)
      fun = CubicSpline(zarr[::-1], Xe[::-1],bc_type='natural')
      z = np.exp(-x) -1.0
      y = np.exp(-3.0 * x / 2.0)
      yre = np.exp(-3.0 * self.x_reion / 2.0)
      dyre = (3.0 / 2.0)*np.exp(-x / 2.0) * (np.exp(-self.delta_x_reion / 2.0) - 1.0 )
      if self.reionization == False and self.helium_reionization == False:
          return fun(z)
          
      elif self.reionization == True and self.helium_reionization == False:
          ival1 = fun(z) + ((1.0 + fHe) /2.0 )*(1.0 + np.tanh((yre - y) / dyre)) 
          return ival1
      
      elif self.reionization == True and self.helium_reionization == True:
          ival2 = fun(z) + ((1.0 + fHe) /2.0)*(1.0 + np.tanh((yre - y) / dyre)) + (fHe / 2.0)*(1.0 + np.tanh((self.z_helium_reion - z) / self.delta_z_helium_reion))
          return ival2
      
  def number_density_electron(self,x):
      """
      

      Parameters
      ----------
      x : x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns 
      -------
      
      n_e = Number density of electron 
      Returns float
      

      """
      ival = self.number_density_Hydrogen(x) * self.Xe(x)
      return ival
  
    

  def optical_depth(self,x):
      """
      

      Parameters
      ----------
      x : x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns 
      -------
      
      tau = optical depth
      Returns float
      

      """
      
      x1 = np.linspace(-50,0,1000)
      vec_H =  np.vectorize(BG.H)
      ival1 = - const.c * const.sigma_T * self.number_density_electron(x1) / vec_H(x1)
      ival2  = CubicSpline(x1, ival1,bc_type='natural')
      ival3 = ival2.integrate(0, x,extrapolate=None)
      return ival3
  
    
  def Der_optical_depth(self,x):
      """
      

      Parameters
      ----------
      x : x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns 
      -------
      
      tau' = Frist derivative of optical depth
      Returns float
      

      """
      
      x1 = np.linspace(-50,0,1000)
      vec_H =  np.vectorize(BG.H)
      ival1 = - const.c * const.sigma_T * self.number_density_electron(x1) / vec_H(x1)
      ival2  = CubicSpline(x1, ival1,bc_type='natural')
      return ival2
  
  def visibility_function(self,x):
      """
      

      Parameters
      ----------
      x : x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns 
      -------
      
      tau = visibility function
      Returns float
      

      """
      x1 = np.linspace(-50,0,1000)
      vec_H =  np.vectorize(BG.H)
      ival1 = - const.c * const.sigma_T * self.number_density_electron(x1) / vec_H(x1)
      ival2  = CubicSpline(x1, ival1,bc_type='natural')
      ival3 = ival2.integrate(0, x,extrapolate=None)
      ival4 = np.exp(- ival3)
      return - ival2(x) * ival4
  
    
  def R(self,x):
      """ Defining R as \frac{4 \Omega_{gamma_0}} {3 \Omega_{b_0} a}
      

      Parameters
      ----------
      x :  Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns
      -------
      Float

      """
      ival = 4.0 * BG.OmegaG0 / (3.0 * BG.OmegaB0 * BG.scale_factor(x))
      return ival
  
    
  def baryon_optical_depth(self,x):
      """
      Parameters
      ----------
      x :  Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns 
      -------
      
      Baryons optical depth 
      Returns float

      """
      x1 = np.linspace(-50,0,1000)
      vec_H =  np.vectorize(BG.H)
      ival1 = - const.c * const.sigma_T * self.R(x1) * self.number_density_electron(x1) / vec_H(x1)
      ival2  = CubicSpline(x1, ival1,bc_type='natural')
      ival3 = ival2.integrate(0, x,extrapolate=None)
      return ival3
  
    
  def z_decoupling(self):
      """
      Defining decoupling redshift when optical depth becomes 1

      Parameters
      ----------
      x :  Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns
      -------
      float

      """
      ival1 = lambda x : self.optical_depth(x) -1.0
      ival = fsolve(ival1 ,[-8])
      return np.exp(- ival) -1.0
  
    
  def z_drag(self):
      """
      Defining Drag redshift when baryon optical depth becomes 1

      Parameters
      ----------
      x :  Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns
      -------
      float

      """
      ival1 = lambda x : self.baryon_optical_depth(x) -1.0
      ival = fsolve(ival1 ,[-8])
      return np.exp(- ival) -1.0
  
    
  def sound_speed_baryon_photon(self,x):
      """
      Sound speed of the photon baryon fluid defined as 
       c \sqrt{\frac{R} {3( 1 + R)}} 

      Parameters
      ----------
      x :  Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns c_s : sound speed for baryon photon fluid
      -------
      float

      """
      ival1 = 3.0 * (self.R(x) + 1.0)
      ival2 = const.c * np.sqrt( self.R(x) /  ival1)
      return ival2
  
    
  def sound_horizon(self):
      """ Sound Horizon : \int_{-\infty}^{x_dec} \frac{c_s}{\mathcal{H}} dx
      

      Parameters
      ----------
      x :  Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns sound Horizon : rs in Mpc
      -------
      Float

      """
      
      x1 = np.linspace(-50,0,1000)
      vec_H =  np.vectorize(BG.Conformal_H)
      ival1 = self.sound_speed_baryon_photon(x1) / vec_H(x1) 
      ival2  = CubicSpline(x1, ival1,bc_type='natural')
      x_dec = -np.log(self.z_decoupling() + 1.0)
      ival3 = ival2.integrate(-50,x_dec ,extrapolate=None)
      return ival3 * constCos.m
  
    
  def Baryon_drag_epoch(self):
      """ Baryon Drag Epoch: \int_{-\infty}^{x_drag} \frac{c_s}{\mathcal{H}} dx
      

      Parameters
      ----------
      x :  Quantity-like ['e-folds'], array-like, or `~numbers.Number`
      Input e-folds.

      Returns sound Horizon : rd in Mpc
      -------
      Float

      """
      
      x1 = np.linspace(-50,0,1000)
      vec_H =  np.vectorize(BG.Conformal_H)
      ival1 = self.sound_speed_baryon_photon(x1) / vec_H(x1) 
      ival2  = CubicSpline(x1, ival1,bc_type='natural')
      x_drag = -np.log(self.z_drag() + 1.0)
      ival3 = ival2.integrate(-50,x_drag ,extrapolate=None)
      return ival3 * constCos.m
  
    


  
