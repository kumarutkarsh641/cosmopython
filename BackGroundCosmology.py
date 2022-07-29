import numpy as np
#from matplotlib import pyplot as plt
#from scipy.interpolate import CubicSpline
from scipy.integrate import quad

from Global import const, constCos

class BackGroundCosmology:
    
    """
     This is a Base class for the cosmology at the background level.
     It holds cosmological paramters and functions releveant for the background.
     
     Input Parameters:
         h          (float); The little Hubble parameter h in H0 = 100h km/s/Mpc
         OmegaB0      (float): Baryonic matter density parameter at z = 0
         OmegaCDM 0   (float): Cold dark matter density parameter at z = 0
         OmegaK0      (float,optional): Curvative density parameter at z = 0
         name        (float,optional): A name for describing the cosmology
         TCMB       (float,optional): The temperature of the CMB today in Kelvin. Fiducial value is 2.725K
         Neff        (float,optional): The effective number of relativistic neutrinos
         delta       (float,optional): The anamolus dimension of Unparticles Cosmology
         y0          (float,optional): the inital unparticles temperature T_0 = 1 + 10**(y0)
    Attributes:    
        OmegaR0      (float): Radiation matter density parameter at z = 0
        OmegaNu0     (float): Massless neutrino density parameter at z = 0
        OmegaM0      (float): Total matter (CDM+b+mnu) density parameter at z = 0
        OmegaK0      (float): Curvature density parameter at z = 0
  
    Functions:
        eta_of_x             (float->float) : Conformal time times c (units of length) as function of x=log(a) 
        H_of_x               (float->float) : Hubble parameter as function of x=log(a) 
        dHdx_of_x            (float->float) : First derivative of hubble parameter as function of x=log(a) 
        Hp_of_x              (float->float) : Conformal hubble parameter H*a as function of x=log(a) 
        dHpdx_of_x           (float->float) : First derivative of conformal hubble parameter as function of x=log(a)
        Tunp_of_x            (float->float) : Evolution of Unparticles temperature as function of x=log(a)     
    """
    # Settings for integration and splines of eta
    




    def __init__(self,h = 0.7, OmegaB0 = 0.046, OmegaCDM0 = 0.224, OmegaK0 = 0.0,w0 = -1.0, TCMB_in_K = 2.725, Neff = 3.046,name = "FiducialCosmology"):
        self.OmegaB0     = OmegaB0
        self.OmegaCDM0   = OmegaCDM0
        self.h           = h
        self.H0          = const.H0_over_h * h
        self.OmegaK0     = OmegaK0
        self.TCMB        = TCMB_in_K * const.K
        self.Neff        = Neff
        self.name        = name
        self.w0          = w0
    
    # compute curvature density
        self.rhoc0      = 3.0*self.H0**(2.0) / (8.0*np.pi*const.G)
        self.OmegaG0    = 2.0* (np.pi**(2.0) / 30.0)*((const.k_b*self.TCMB)**(4.0) / (const.hbar**(3.0)*const.c**(5.0))) / self.rhoc0
        self.OmegaNu0   = (7.0 / 8.0)*(4.0 / 11.0)**(4.0 / 3.0)*self.Neff*self.OmegaG0
        self.OmegaR0    = self.OmegaG0 + self.OmegaNu0
        self.OmegaM0    = self.OmegaB0 + self.OmegaCDM0
        self.OmegaDE0   = 1.0 - self.OmegaM0 -self.OmegaR0 -self.OmegaK0
        self.Tnu0       = 0.7137658555036082*self.TCMB
        self.Hubble_time = const.seconds_to_Gyr(1.0 / self.H0)
        self.Hubble_distance = (const.c / self.H0) * constCos.m
        
    #properties
        @property
        def is_flat(self):
            """ Return bool; 'True" if Cosmology is flat. """
            return bool((self.OmegaK0== 0.0) and (self.OmegaT0 == 1.0))
    
        @property 
        def OmegaT0(self):
            """ Omega total : the total energy density / critical density at z = 0."""
            return self.OmegaM0 + self.OmegaR0 + self.OmegaDE0 + self.OmegaK0
    
        @property 
        def OmegaDE0(self):
            """ Omega Dark energy: dark energy density / crictical energy density at z = 0"""
            return self.OmegaDE0
        @property 
        def Tnu0(self):
            """ Temperature of the neutrino background  at z=0."""
            return self.Tnu0
    
        @property 
        def Hubble_time(self):
            """ Hubble time expressed in Gyr"""
            return self.Hubble_time
    
        @property 
        def Hubble_distance(self):
            """ Hubble distance expressed in Mpc"""
            return self.Hubble_distance
    
        @property 
        def critical_density0(self):
            """ Crtical Energy density at redshift x = 0 i.e. present day"""
            return self.rhoc0
    
        @property
        def Omega_gamma0(self):
            """ Omega gamma at x = 0 """
            return self.OmegaG0
    
        @property 
        def Omega_nu0(self):
            """ Omega neutrino at x = 0 """
            return self.OmegaNu0
    
        @property 
        def Omega_Rad0(self):
            """ Omega radiaton at x = 0 ; Omega_nu0 + Omega_gamma0 """
            return self.OmegaR0
    
        @property 
        def Omega_M0(self):
            """ Omega matter at x = 0 ; Omega_B0 + Omega_CDM0 """
            return self.OmegaM0
    
    # ---------------------------------
    def w(self,x):
            r"""The dark energy equation of state.
        
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        
        Returns
        -------
        w : ndarray or float
            The dark energy equation of state.
            `float` if scalar input.
        
        
        Notes
        -----
        The dark energy equation of state is defined as
        :math:`w(x) = P(x)/\rho(x)`, where :math:`P(x)` is the pressure at
        efolds x and :math:`\rho(x)` is the density at e-folds x, both in
        units where c=1.
        This must be overridden by subclasses.
        """
        
            return self.w0

    def Omega_B(self,x):
        """The baryonic density parameter at redshift ``x``.
        
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        Returns
        -------
        Omega_B : ndarray or float
            The Baryonic density relative to the critical density at each redshift.
            Returns float if input scalar.
        """
        return self.OmegaB0*np.exp(-3.0*x) * self.inv_efunc(x) **(2.0)
    
    def Omega_CDM(self,x):
        """The Cold Dark matter density parameter at redshift ``x``.
        
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        Returns
        -------
        Omega_CDM : ndarray or float
            The Cold Dark Matter density relative to the critical density at each e-folds.
            Returns float if input scalar.
        """
        return self.OmegaCDM0*np.exp(-3.0*x) * self.inv_efunc(x) **(2.0)
    
    def Omega_Gamma(self,x):
        """The Photon density parameter at efolds ``x``.
        
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        Returns
        -------
        Omega_B : ndarray or float
            The photon density relative to the critical density at each e-fold.
            Returns float if input scalar.
        """
        return self.OmegaG0*np.exp(-4.0*x) * self.inv_efunc(x) **(2.0)
    
    def Temp_to_x(self,x):
        """
        

        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.

        Returns
        -------
        Temp : ndarray or float
            The Temp at each e-fold.
            Returns float if input scalar.

        """
        return self.TCMB * np.exp(-x)
    
    def Omega_Nu(self,x):
        """The Neutrino density parameter at efolds ``x``.
        
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        Returns
        -------
        Omega_B : ndarray or float
            The Neutrino density relative to the critical density at each e-fold.
            Returns float if input scalar.
        """
        return self.OmegaNu0*np.exp(-4.0*x) * self.inv_efunc(x) **(2.0)
    
    def Omega_K(self,x):
        """The Curvature density parameter at  efolds ``x``.
        
        Parameters
        ----------
        z : Quantity-like ['efolds'], array-like, or `~numbers.Number`
            Input efolds.
        Returns
        -------
        Ok : ndarray or float
            The Curvature density relative to the critical density at each efolds.
            Returns float if input scalar.
        """
        return self.OmegaK0*np.exp(-2.0*x) * self.inv_efunc(x) **(2.0)
    
    def Omega_R(self,x):
        
        """The Total radiation energy density parameter at  efolds ``x``.
        
        Parameters
        ----------
        x : Quantity-like ['efolds'], array-like, or `~numbers.Number`
            Input efolds.
        Returns
        -------
        OR : ndarray or float
            The Total energy density relative to the critical density at each efolds.
            Returns float if input scalar.
        """
        return np.exp(-4.0*x) * (self.OmegaNu0 + self.OmegaG0) * self.inv_efunc(x) **(2.0)
    
    def Omega_M(self,x):
        
        """The Total matter energy density parameter at  efolds ``x``.
        
        Parameters
        ----------
        x : Quantity-like ['efolds'], array-like, or `~numbers.Number`
            Input efolds.
        Returns
        -------
        Om : ndarray or float
            The total matter density relative to the critical density at each efolds.
            Returns float if input scalar.
        """
        return np.exp(-3.0*x) * (self.OmegaB0 + self.OmegaCDM0) * self.inv_efunc(x) **(2.0)
        
    
    def Omega_DE(self,x):
        """ Return the denisity parameter for dakr energy at e-folds x.
        
        Parameters 
        ----------
        x : Quantity-like ['efolds'], array-like, or `~numbers.Number`
            Input efolds.
        Returns
        -------
        Omega_DE : ndarray or float
            The Curvature density relative to the critical density at each efolds.
            Returns float if input scalar.
        """
        return self.OmegaDE0*self.DE_density_scale(x) * self.inv_efunc(x) **(2.0)
    
    
    
    
        
    def Omega_Tot(self,x):
        """The total density parameter at efolds ``z``.
        
        Parameters
        ----------
        x : Quantity-like ['efolds'], array-like, or `~numbers.Number`
            Input efolds.
        Returns
        -------
        Otot : ndarray or float
            The total density relative to the critical density at each efolds.
            Returns float if input scalar.
        """
        return self.Omega_B(x) + self.Omega_CDM(x) + self.Omega_Gamma(x) + self.Omega_Nu(x) + self.Omega_DE(x) + self.Omega_K(x)
    
    def w_integrand(self,x):
        
        """Internal convenience function for w(x) integral (eq. 5 of [1]_).
        Parameters
        ----------
        ln1pz : `~numbers.Number` or scalar ndarray
            Assumes scalar input, since this should only be called inside an
            integral.
        References
        ----------
        .. [1] Linder, E. (2003). Exploring the Expansion History of the
               Universe. Phys. Rev. Lett., 90, 091301.
        """
        return 1.0 + self.w(x)
    
    def DE_density_scale(self, x):
        
        r"""Evaluates the e-folds dependence of the dark energy density.
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        Returns
        -------
        I : ndarray or float
            The scaling of the energy density of dark energy with e-folds.
            Returns `float` if the input is scalar.
        Notes
        -----
        The scaling factor, I, is defined by :math:`\rho(z) = \rho_0 I`,
        and is given by
        .. math::
           I = \exp \left( 3 \int_{a}^1 \frac{ da^{\prime} }{ a^{\prime} }
                          \left[ 1 + w\left( a^{\prime} \right) \right] \right)
        The actual integral used is rewritten from [1]_ to be in terms of z.
        It will generally helpful for subclasses to overload this method if
        the integral can be done analytically for the particular dark
        energy equation of state that they implement.
        References
        ----------
        .. [1] Linder, E. (2003). Exploring the Expansion History of the
               Universe. Phys. Rev. Lett., 90, 091301.
        """
        # This allows for an arbitrary w(z) following eq (5) of
        # Linder 2003, PRL 90, 91301.  The code here evaluates
        # the integral numerically.  However, most popular
        # forms of w(z) are designed to make this integral analytic,
        # so it is probably a good idea for subclasses to overload this
        # method if an analytic form is available.
        ival = quad(self.w_integrand,0,x)[0]
        #ival = solve_ivp(self.w_integrand,[0,-50],[0],method='RK45',t_eval=[x])
        return np.exp(-3.0 * ival)
    
    def efunc(self, x):
        """Function used to calculate H(x), the Hubble parameter.
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        Returns
        -------
        E : ndarray or float
            The e-folds scaling of the Hubble constant.
            Returns `float` if the input is scalar.
            Defined such that :math:`H(x) = H_0 E(x)`.
        Notes
        -----
        It is not necessary to override this method, but if de_density_scale
        takes a particularly simple form, it may be advantageous to.
        """
        x1 = np.exp(-x)
        return np.sqrt((self.OmegaB0 + self.OmegaCDM0) * x1 **(3.0) + self.OmegaR0 * x1 **(4.0) + self.OmegaK0 * x1 **(2.0) + self.OmegaDE0 * self.DE_density_scale(x))

    def inv_efunc(self, x):
        """Inverse of ``efunc``.
        Parameters
        ----------
        x : Quantity-like ['e-folds'], array-like, or `~numbers.Number`
            Input e-folds.
        Returns
        -------
        E : ndarray or float
            The redshift scaling of the inverse Hubble constant.
            Returns `float` if the input is scalar.
        """
        # Avoid the function overhead by repeating code
        x1 = np.exp(-x)
        return ((self.OmegaB0 + self.OmegaCDM0) * x1 **(3.0) + self.OmegaR0 * x1 **(4.0) + self.OmegaK0 * x1 **(2.0) + self.OmegaDE0 * self.DE_density_scale(x)) **(-0.5)
        

    def H(self,x):
        """ Hubble parameter (km/s/Mpc) at e-fold x
        
        Parameters
        
        - - - - - - - - 
        
        x : Quantity -like [e-fold] Input e-fold
        
        Returns 
        
        --------------
        
        H : Hubble parameter at each input e-fold
        
        """
        
        return self.H0 * self.efunc(x)
    
    def Conformal_H(self,x):
        """ Conformal Hubble parameter (km/s/Mpc) at e-fold x
        
        Parameters
        
        - - - - - - - - 
        
        x : Quantity -like [e-fold] Input e-fold
        
        Returns 
        
        --------------
        
        H : Conformal Hubble parameter at each input e-fold
        
        """
        
        return self.H0 * self.efunc(x) * self.scale_factor(x)
    
    def scale_factor(self,x):
        
        """ Scale factor at e -folds ``x``
        
        The scale factor is defined as :math: a = exp(x)
        
        Parameters
        
        -----------------
        
        x : Quantity -like [e-fold] Input e-fold
        
        Returns 
        
        a:  float
            Scale factor at each input e-folds
        """
        return np.exp(x)
    
    def eta_of_x(self,x):
        """
        \eta(x) = \int_{-\infty}^{x} \frac{c}{\mathcal{H}} dx

        Parameters
        ----------
        x : Quantity -like [e-fold] Input e-fold

        Returns \eta (x) 
        -------
        float

        """
        Y1 = lambda x: const.c / self.Conformal_H(x)
        ival = quad(Y1,- 50.0 , x)[0]
        return ival * constCos.m
        
    
    def age_Universe(self):
        """
        Age of Universe in GYrs

        

        Returns Age of Universe
        -------
        

        """
        ival = quad(self.inv_efunc,-50,0)
        _age = self.Hubble_time * ival[0]
        return _age
    
    def z_eq_rad_matter(self):
        """
        redshift at matter radiation equality

        Returns redshifts float
        -------
        None.

        """
        ival = self.OmegaM0 / self.OmegaR0  -1.0
        return ival
    
    def k_eq_rad_matter(self):
        """
        wavenumber at matter-radiation equality

        Returns k_eq
        -------
        float

        """
        ival = - np.log(1.0 + self.z_eq_rad_matter())
        return 1.0 / self.eta_of_x(ival)

    
    
        
        
        
        

    

    
    
    
    
        
        

    

     
    
    
    
    
        
        
    
    
    

    
    
    






