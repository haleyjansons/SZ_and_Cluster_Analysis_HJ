""" 
Tools for computing Pressure Profiles, Aperture Photometry, Submap plotting, etc based on DeepSZSim and Battaglia 2012.
Example usage and explanations are in SZ_Simulation_Guide.ipynb and ACT_Cluster_Aperture_Photometry.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm, SymLogNorm
import pandas as pd
from pixell import enmap, utils, enplot
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from colossus.halo import mass_adv
from colossus.cosmology import cosmology

#define cosmological values based on Battaglia2012

cosmo_battaglia = FlatLambdaCDM(
    H0 = 70 * u.km / u.s / u.Mpc,
    Om0 = 0.25
    # flat ΛCDM ⇒ ΩΛ = 1 − Ωm = 0.75
)

#need to make colossus cosmology match Battaglia ones we defined
cosmo_params = {
    'flat': True,
    'H0': cosmo_battaglia.H0.value,  # 70.0 (km/s/Mpc)
    'Om0': cosmo_battaglia.Om0,      # 0.25
    'Ob0': 0.043,
    'sigma8': 0.8,  
    'ns': 0.96,
}
cosmology.addCosmology('battaglia', cosmo_params)
cosmology.setCosmology('battaglia')


# constants
keVcm3_to_Jm3 = ((1 * u.keV / (u.cm**3.)).to(u.J / (u.m**3.))).value
Mpc_to_m = (1 * u.Mpc).to(u.m).value
Thomson_scale = (c.sigma_T/(c.m_e * c.c**2)).value

# conversion factor from Battaglia 2012
thermal_to_electron_pressure = 1 / 1.932 

def galaxy_submap(file_path, ra, dec, radius): # input ra and dec in degrees, radius in arcmin

  ra = ra * utils.degree # converts to radians
  dec = dec * utils.degree # converts to radians
  radius = radius * utils.arcmin # converts to radians

  box = np.array([[dec - radius, ra - radius], [dec + radius, ra + radius]])
  imap_box = enmap.read_map(file_path, box=box)

  return imap_box


def get_tSZ_signal_aperture_photometry(dT_map, radmax_arcmin, pixel_scale, 
                                       fmax=np.sqrt(2)):
    """
    Retrieve aperture photometry of the tSZ signal
    
    Parameters:
    ----------
    dT_map: float array
        map in uK
    radmax_arcmin: float
        radius of the inner aperture in arcmin
    pixel_scale: float
        arcmin per pixel for the current settings
    fmax: float
        fmax * radmax_arcmin is the radius of the outer aperture in arcmin

    Returns:
    -------
    disk_mean: float
        average value within an annulus of inner radius r
    ring_mean: float
        average value within an annulus of outer radius sqrt(2)*r
    tSZ signal: float
        thermal SZ effect signal
    """

    radmax_pixels = radmax_arcmin / pixel_scale
    radius_out_pixels = radmax_pixels * fmax
    
    center = np.array(dT_map.shape) // 2
    x, y = np.indices(dT_map.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    #take the mean in inner disk
    disk_mean = dT_map[r < radmax_pixels].mean()
    
    #take mean of outer annulus
    ring_mean = dT_map[(r >= radmax_pixels) & (r < radius_out_pixels)].mean()
    
    #signal comes from the difference between the means
    tSZ = disk_mean - ring_mean
    
    return disk_mean, ring_mean, tSZ


#convert M500c to M200c (M500c is the format of Catalog and M200c is format for our equations)
def M500c_to_M200c(M500c_cat, z):
    """
    Convert M500c (in Msun, not Msun/h) at redshift z
    to M200c (in Msun) using Colossus and a c(M,z) model.
    """
    h = cosmo_battaglia.h

    # Convert Catalog format of 1e14*Msun → Msun/h (needed for Colossus)
    M500c = M500c_cat * 1e14 * h

    # Do the conversion at fixed cosmology
    M200c, R200c, c200c = mass_adv.changeMassDefinitionCModel(
        M500c, z, '500c', '200c'
    )

    # Convert back to physical Msun and Mpc
    M200c_Msun = M200c / h             # Msun

    return M200c_Msun

def get_critical_density(redshift_z):
    """
    Calculates critical density beased on redshift value
    
    Parameters:
    -----------
    redshift
    
    Returns:
    ----------
    Critical_density as calculated in Battaglia 
    
    """
    
    # Critical density at z
    rho_cr = cosmo_battaglia.critical_density(redshift_z) # → kg/m^3 by default
    rho_cr_Msun_Mpc3 = rho_cr.to('Msun / Mpc3')
    
    return rho_cr_Msun_Mpc3
    

def get_R200(M200_SM, redshift_z):
    """
    
    Parameters:
    -----------
    M200_SM: float
         mass contained within R200, in units of solar masses
    redshift_z: float
         redshift of the cluster (unitless)
    
    Returns:
    --------
    R200_Mpc: instance
        radius of the cluster at 200 times the critical density of the universe in Mpc
    """

    rho_cr = get_critical_density(redshift_z)
    
    M200 = M200_SM * u.Msun                     # Msun
    R200_Mpc = (3 * M200 / (4 * np.pi * 200 * rho_cr))**(1/3)
    
    return R200_Mpc
    

def get_P200(M200_SM, redshift_z):
    """
    Calculates the P200 normalized thermal pressure of a cluster, as defined in Battaglia 2012.

    Parameters:
    -----------
    M200_SM: float
         mass contained within R200, in units of solar masses
    redshift_z: float
         redshift of the cluster (unitless)
    
   
    Returns:
    --------
    P200_kevcm3: instance
        thermal pressure of the shell defined by R200 in units of keV/cm**3
    """
    
    Omega_b = 0.043
    Omega_m = 0.25
    rho_cr = get_critical_density(redshift_z)
    
    GM200 = c.G * M200_SM * u.Msun * 200. * rho_cr #The Equation from above!! #R_Delta = 200
    
    R200_Mpc = get_R200(M200_SM, redshift_z)
    
    fbR200 = (Omega_b / Omega_m) / (2. * R200_Mpc)  # From Battaglia2012
   
    P200 = GM200 * fbR200
    P200_kevcm3 = P200.to(u.keV / u.cm**3.)  # Unit conversion to keV/cm^3
    
    return (P200_kevcm3)
    

def get_param(A0, alpha_m, alpha_z, M200_SM, redshift_z):
    
    """
    Calculates independent params using the formula from Battaglia 2012, Equation 11
    in order for use in the pressure profile defined in Equation 10

    Parameters:
    -----------
    A0: float
        normalization factor
    alpha_m: float
        power law index for the mass-dependent part of the function
    alpha_z: float
        power law index for the redshift dependent part
    M200_SM: float
        mass of the cluster at 200 times the critical density of the 
        universe in units of solar masses
    redshift_z: float
        redshift of the cluster

    Returns:
    --------
    A: float
        formula from Battaglia 2012, Eq 11
    """
    
    A = A0 * (M200_SM / 1e14)**alpha_m * (1. + redshift_z)**alpha_z
    
    return (A)
    

def get_P0(M200_SM, redshift_z):
    """
    Calculates P0, the normalization factor/amplitude, 
    from Battaglia 2012, using the values from Table 1

    Parameters:
    -----------
    M200_SM: float
        mass of the cluster at 200 times the critical density of the 
        universe, in units of solar masses
    redshift_z: float
        redshift of the cluster (unitless)

    Returns:
    -------
    P0: float
        normalization factor for the Battaglia 2012 profile

    """
    return get_param(18.1, 0.154, -0.758, M200_SM, redshift_z)
    


def get_xc(M200_SM, redshift_z):
    """
    Calculates xc (the core-scale factor) from Battaglia 2012 Table 1

    Parameters:
    -----------
    M200_SM: float
        mass of the cluster at 200 times the critical density of the 
        universe, in units of solar masses
    redshift_z: float
        redshift of the cluster (unitless)

    Returns:
    --------
    xc: float
        core-scale factor for the Battaglia 2012 profile

    """
    return get_param(0.497, -0.00865, 0.731, M200_SM, redshift_z)
    

def get_beta(M200_SM, redshift_z):
    
    """
    Calculates beta, the power law index, from Battaglia 2012 from Table 1.

    Parameters:
    ----------
    M200_SM: float
        mass of the cluster at 200 times the critical density of the 
        universe, in units of solar masses
    redshift_z: float
        redshift of the cluster (unitless)

    Returns:
    -------
    beta: float
        power law index for the Battaglia 2012 profile

    """
    return get_param(4.35, 0.0393, 0.415, M200_SM, redshift_z)


def get_Pfit(radius_mpc, M200_SM, redshift_z,
                    alpha = 1.0, gamma = -0.3):
    """
    Calculates the Pfit profile using the Battaglia profile, Battaglia 2012,
    Equation 10. It is unitless (normalized by P200)-- meaning it's really Pfit/P200.

    Parameters:
    -----------
    radius_mpc: float or float array
        radius for the pressure to be calculated at, in units of Mpc
    M200_SM: float
        mass contained within R200, in units of solar masses
    R_200_Mpc: float
        radius of the cluster at 200 times the critical density of the universe in Mpc
    redshift_z: float
        redshift of the cluster (unitless)
    alpha: float
        fixed by Battaglia et al 2012 to 1.0
    gamma: float
        fixed by Battaglia et al 2012 to -0.3

    Returns:
    --------
    Pfit: float or float array
        thermal pressure profile normalized by P200, which has units keV/cm**3
    """
    
    if isinstance(radius_mpc, Quantity):
        rvals = radius_mpc.to_value(u.Mpc)   # strip units → float
    else:
        rvals = radius_mpc                   # already floats
    
    #these vars are calculated using a function of mass and redshift
    P0 = get_P0(M200_SM, redshift_z)
    xc = get_xc(M200_SM, redshift_z)
    beta = get_beta(M200_SM, redshift_z)
    
    R200 = get_R200(M200_SM, redshift_z).to_value(u.Mpc)  # float
    P200 = get_P200(M200_SM, redshift_z)
    
    x =  rvals / R200
    
    Pfit = P0 * (x / xc)**gamma * (1 + (x / xc)**alpha)**(-beta)
    
    return (Pfit)


def arcmin_to_mpc(r_array, redshift_z):
    # attach unit
    r_array_arcmin = r_array * u.arcmin
    
    # convert to radians
    theta = r_array_arcmin.to(u.rad)
    
    # Make theta dimensionless (remove the 'rad' label)
    theta = theta.value * u.dimensionless_unscaled
    
    # get angular diameter distance (Quantity in Mpc)
    DA = cosmo_battaglia.angular_diameter_distance(redshift_z) 
    
    # multiply to get physical radius (Quantity in Mpc)
    r_array_mpc = (theta * DA).to(u.Mpc)
    
    return r_array_mpc

def Pe_to_y(radii_mpc, M200_SM, redshift_z, alpha = 1.0, gamma = -0.3):
    """
    Converts from an electron pressure profile to a compton-y profile;
    integrates over line of sight from -1 to 1 Mpc relative to center.

    Parameters:
    -----------
    radii_mpc: array
        array of radii corresponding to the profile in Mpc
    M200_SM: float
        mass contained within R200, in units of solar masses
    redshift_z: float
        redshift of the cluster (unitless)
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3

    Returns:
    --------
    y_pro: array
        Compton-y profile corresponding to the radii.
        
    """
    
    #convert all Mpc to float values:
    R200 = get_R200(M200_SM, redshift_z).to_value(u.Mpc)  # float
    P200 = get_P200(M200_SM, redshift_z).value
    
    if isinstance(radii_mpc, Quantity):
        rvals = radii_mpc.to_value(u.Mpc)   # strip units → float
    else:
        rvals = radii_mpc                   # already floats
    
    pressure_integ = np.empty_like(rvals)
    rmax = rvals.max()
    
    for i, radius in enumerate(rvals):
        # Multiply profile by P200 specifically for Battaglia 2012 profile,
        # since it returns Pfit/P200 instead of Pfit
        rv = radius
        if (rv >= R200):
            pressure_integ[i] = 0
            
        else:
            l_mpc = np.linspace(0, np.sqrt(rmax**2. - rv**2.) + 1., 1000)  # Get line of sight axis
            th_pressure = get_Pfit(np.sqrt(l_mpc**2 + rv**2), M200_SM, redshift_z, alpha = alpha,
                                  gamma = gamma)
            
            integral = np.trapz(th_pressure, l_mpc)
            pressure_integ[i] = integral
    
    y_pro = pressure_integ * P200 * keVcm3_to_Jm3 * Thomson_scale * thermal_to_electron_pressure * 2 * Mpc_to_m
    
    return y_pro


def make_y_submap(M200_SM, redshift_z, image_size_pixels, pixel_size_arcmin, alpha = 1.0,
                   gamma = -0.3):
    """
    Converts from an electron pressure profile to a compton-y profile,
    integrates over line of sight from -1 to 1 Mpc relative to center.

    Parameters:
    -----------
    M200_SM: float
        mass contained in R200, in units of solar masses
    redshift_z: float
        redshift of the cluster (unitless)
    image_size_pixels: int
        size of final submap in number of pixels
    pixel_size_arcmin: float
        size of each pixel in arcmin
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3

    Returns:
    -------
    y_map: array
        Compton-y submap with shape (image_size_pixels, image_size_pixels)
    """

    #Build a 1-D radius grid 
    #X runs from 0 to half the map in arcmin, sampled at pixel centers
    X = np.linspace(0, (image_size_pixels // 2) * pixel_size_arcmin, image_size_pixels//2 + 1)
    
    #convert to Mpc but remove the unit so X is a float
    X = arcmin_to_mpc(X, redshift_z).to_value(u.Mpc) 
    
    #Avoid singularity at center by setting minimum radius > 0
    min_radius = arcmin_to_mpc(pixel_size_arcmin*0.1, redshift_z).to_value(u.Mpc) 
    
    #Forms a grid of projected radii from the 1-D array X with the Square root part of the equation
    R = np.maximum(min_radius, np.sqrt(X[:, None]**2 + X[None, :]**2).flatten()) # [Mpc], 1D
    
    #evaluate compton-y for each neccesary radius 
    compton_y = Pe_to_y(R, M200_SM, redshift_z, alpha = alpha, gamma = gamma)  
    
    # Build empty map
    n = X.size
    y_map = np.zeros((2*n - 1, 2*n - 1))
    
    # Fill upper-right quadrant and mirror
    for i, x in enumerate(X):
        for j in range(i, len(X)):
            y = X[j]
            r_here = max(min_radius, np.sqrt(x**2 + y**2))
            idx = np.where(np.isclose(R, r_here, atol=1e-10, rtol=1e-10))[0][0]
            ijval = compton_y[idx]
            
            y_map[n + i - 1, n + j - 1] = ijval
            if j != i:
                y_map[n + j - 1, n + i - 1] = ijval
            
    # Mirror vertically
    for i in range(n):
        y_map[n - i - 1] = y_map[n + i - 1]

    # Mirror horizontally
    for j in range(n):
        y_map[:, n - j - 1] = y_map[:, n + j - 1]

    return y_map

def f_sz(freq_ghz, T_CMB_K):
    """
    leading order correction to blackbody from Compton scattering
    
    Parameters:
    ----------
    freq_ghz: float
        Observation frequency f, in units of GHz
    T_CMB_K: instance of temperature spectrum
        Temperature of CMB in K

    Returns:
    ------
    fsz: float
        radiation frequency
    """
    #Takes input in units of GHz
    f=freq_ghz*u.GHz 

    #Unit conversion
    f=f.to(1/u.s) 
    x = (c.h * f / c.k_B / T_CMB_K).value
    fsz = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

    return fsz

def plot_graphs(image, title, xlabel, ylabel, cbarlabel, width,
                logNorm = False):
    """
    Plotting tool function for our 2D submaps and CMB maps 
    
    Parameters:
    -----------
    image: float array
        graph that is plotted
    title: str
        title of the graph
    xlabel: str
        label of the x-axis
    ylabel: str
        label of the y-axis
    cbarlabel: str
        label of the color bar
    width: int
        half rounded down of the width of output plot in pixels (eg, image size = 2*width+1)
    logNorm: bool
        if true, uses a logarithmic normalization for the plot (using SymLogNorm in case values are negative)
    

    Returns:
    -------
    none
    """
    
    if logNorm:
        if np.min(image)<0:
            imgflatabs = np.abs(image.flatten())
            im = plt.imshow(image, norm = SymLogNorm(linthresh =  np.min(imgflatabs[np.nonzero(imgflatabs)])))
        else:
            im = plt.imshow(image, norm=LogNorm())
    else:
        im = plt.imshow(image)
        
    cbar = plt.colorbar(im)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar.set_label(cbarlabel, rotation=270)

    return im, cbar


