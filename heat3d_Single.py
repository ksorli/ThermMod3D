#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:15:22 2023

@author: kyso3185

Complement for binary systems is heat3d_binary_shadowing_vectorized.py

Full single body thermal model, using shadowing, view factors and obliquity 
built off of vectorized thermal model. 

Shadows and view factors are precalculated and fed in. 


"""


# Physical constants:
sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
solarConstant = 1361.0 # Solar constant at 1 AU [W.m-2]
chi = 2.7 # Radiative conductivity parameter [Mitchell and de Pater, 1994]
R350 = chi/350**3 # Useful form of the radiative conductivity
TWOPI = 6.283185307

# Numerical parameters:
F = 0.1 # Fourier Mesh Number, must be <= 0.5 for stability
m = 10 # Number of layers in upper skin depth [default: 10]
n = 5 # Layer increase with depth: dz[i] = dz[i-1]*(1+1/n) [default: 5]
b = 20 # Number of skin depths to bottom layer [default: 20]

# Accuracy of temperature calculations
# The model will run until the change in temperature of the bottom layer
# is less than DTBOT over one diurnal cycle
DTSURF = 0.1 # surface temperature accuracy [K]
DTBOT = DTSURF # bottom layer temperature accuracy [K]
NYEARSEQ = 1.0 # equilibration time [orbits]
NPERDAY = 24 # minimum number of time steps per diurnal cycle

# NumPy is needed for various math operations
import numpy as np

# MatPlotLib and Pyplot are used for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Methods for calculating solar angles from orbits
import orbits

# Planets database
import planets

# Read in Shape Model
#import shapeModel as shape
import shapeModelMultiples as shape


# For interpolation functions
import scipy.interpolate as interp
import scipy.sparse as sparse

# For interpreting facets  
import kdtree_3d_Kya as kdtree_3d


# Automatic view factor calculation 
import visibilityFunctions as vis

# #Multiprocessing and speedups
import multiprocessing
from itertools import repeat
from numba import jit
#
from multiprocessing import Manager
import pathos
mp = pathos.helpers.mp

# Timing
import time


#Set up solar values and physical constants 
sigma = 5.67051196e-8  # Stefan-Boltzmann Constant [W m^-2 K^-4]
solarLum = (3.826e26 / (4.0*np.pi)) #Solar luminosity [W]
solar = 3.826e26 / (4.0*np.pi)
s0 = solarLum  #[W.m-2]  



        



##############################################################################
## Single Body Model Class
##############################################################################
class SingleModel(object):
    def __init__(self, shapeModelPath, planet = planets.Moon, ndays=1,nyears = 1.0,local = False,shadowLookupPath = None, vfPath = None):
    
        # Change Equilibration Time
        NYEARSEQ = nyears   
        print ("Equilibration time: {} years".format(NYEARSEQ))
    
        # Initialize
        self.planet = planet # Set planet entry from planets.py 
        print ("Selected Planet: {}".format(planet.name))
        self.Sabs = self.planet.S * (1.0 - self.planet.albedo)
        self.r = self.planet.rAU # solar distance [AU]
        self.nu = np.float() # orbital true anomaly [rad]
        self.nudot = np.float() # rate of change of true anomaly [rad/s]
        self.dec = np.float() # solar declination [rad]
        self.solarFlux = np.float() # Solar flux (W/m^2) at the body given the distance to the sun
        #self.sPos = np.array([0,0,0])
       
        # Read in Shape Model & initialize corresponding values
        self.local = local
        self.shape = shape.shapeModel(shapeModelPath,local)
        self.facetNum = self.shape.facetNum
        print ("Shape Model Loaded")
        
        # Initialize arrays for latitudes, fluxes, shadows 
        self.lats = self.shape.lats#np.zeros(self.facetNum) # np array of latitudes for each facet
        
        # Rotation of the body about own axis 
        # Used for shadowing and view factor setup  
        self.theta = np.float() # Radians, starts at 0.
        
        
        # Load shadow array
        shadowArr = np.load(shadowLookupPath)
        shadowArr[shadowArr <= 0] =  0.
        self.shadows = shadowArr

        print ("Shadow array loaded")
        
        
        # Initialize profile(s)
        self.profile = profile(planet = self.planet, facetNum = self.facetNum, lat = self.lats, emis = self.planet.emissivity)
    
    
        # Model run times
        # Equilibration time -- TODO: change to convergence check
        self.equiltime = NYEARSEQ*planet.year - \
                        (NYEARSEQ*planet.year)%planet.day
        # Runtime for output
        self.endtime = self.equiltime + ndays*planet.day
        self.t = 0.
        
        
        # Get Timestep 
        # Uses mean temperature
        self.dt = getTimeStep(self.profile, planet.day)
        
        # Alternatively, use max temperature at perihelion (more important for very hot bodies that require small timesteps)
        # Get conductivity associated with max temperatures to set timestep 
        # maxTemp = getPerihelionSolarIntensity(planet)
        # maxTempK = thermCond(self.profile.kc, maxTemp)
        #self.dt = getHighTempTimeStep(self.profile, maxTempK, planet.day)
        
        # Set timestep for readout
        self.dtout = self.dt
        
        # Check for maximum time step
        dtmax = self.planet.day/NPERDAY
        if self.dt > dtmax:
            self.dtout = dtmax
        
        # Array for output temperatures and local times
        N_steps = np.int((ndays*planet.day)/self.dtout)
        self.N_steps = N_steps
        print ("Timesteps: {}".format(self.N_steps) )
        
        #  Layers
        N_z = np.shape(self.profile.z)[0]
        self.N_z = N_z
        

        # Temperature and local time arrays 
        self.T = np.zeros([N_steps, N_z,self.facetNum])
        self.lt = np.zeros((N_steps,3)) #[N_steps, 3]) # Time, theta, nu
        self.readoutScat = np.zeros([N_steps, self.facetNum])


        # Resolution for view factors and shadowing 
        # count how many facets around equator (ie have lat of ~0?) 
        # This needs to be split into two for a binary 
        if shadowLookupPath == None:
            print ("Calculating facet resolution around equator")
            lats = np.zeros(self.facetNum)
            longs = np.zeros(self.facetNum)
            equatorial_facets = 0
            for tri in self.shape.tris:
                lats[tri.num] = tri.lat
                longs[tri.num] = tri.long
                if tri.lat < 5 and tri.lat >= 0:
                    equatorial_facets += 1
                    
            
            # Find how frequently you need to update shadowing and view factors 
            self.deg_per_facet = 360 / equatorial_facets # degrees per facet
            print ("Equatorial facets: "+str(equatorial_facets))
            print ("Degrees per facet: "+str(self.deg_per_facet))
            
        else: 
            print ("Shadowing Lookup Table provided. Skipping resolution check")
        
        
        
        # View Factors
        if vfPath != None:
            if vfPath.endswith('.npy'):
                self.viewFactors = np.load(vfPath) 
                #self.IncludeReflections = True
            elif vfPath.endswith('.npz'):
                vf = sparse.load_npz(vfPath)
                self.viewFactors = np.asarray(vf.toarray())
                print (np.shape(self.viewFactors))
                #self.IncludeReflections = True 
            
        else: # If automatically calculating view factors 
            
            print ("No view factor file provided. Automatically calculating for provided shape")
            
            ##############################################################################
            ## Multiprocessed model 
            ## DO NOT ALTER
            ##############################################################################
            # local = False
            # shape = shape.shapeModel(shapeModelPath,local)
            # facets = len(shape.tris)
            
            # KD Tree
            KDTree = kdtree_3d.KDTree(self.shape.tris,leaf_size = 5, max_depth = 0)
            
            
            # Calculating view factors
            # Establish multiprocessing manager and thread safe storage
            manager = Manager()
            
            iVals = np.arange(0,self.facetNum)
            viewFactorStorage = manager.dict()
            
            visStartTime = time.perf_counter()
            
            p = mp.Pool(multiprocessing.cpu_count())
            viewFactors = np.zeros((self.facetNum,self.facetNum))
            viewFactorRun = p.starmap(vis.viewFactorsHelper,zip(repeat(self.shape.tris),repeat(self.facetNum),repeat(KDTree),iVals,repeat(viewFactorStorage),repeat(local)))
            
            
            for i in iVals:
                viewFactors[i,:] = viewFactorStorage[i]
            
            print ("Visibility calculations took: "+str(time.perf_counter()-visStartTime)+" sec")
            print ("Number of final non zero view factors: "+str(np.count_nonzero(viewFactors)))
        
                
            self.viewFactors = viewFactors
            
        # else: # If using null (zero) view factors
        
        #     print ("WARNING: AUTOMATIC VIEW FACTORS NOT ENABLED! \n Proceeding with null view factors")
            
        
                
        # Sun location initialization 
        self.sLoc = np.asarray([0,0,self.planet.rsm])#np.asarray([0,0,1.496e11]) # 1 AU # make this self.sLoc
        self.newSLoc = self.sLoc
        self.baseDist = np.float()
        
        print ("Completed Visibility Setup")
        
        self.vfCount = 0
        
        
        
    def run(self,endTrueAnomaly = 0., calcShadows = False, vfFreq = 10):

        
        # Might need to update shadows in a regular, but infrequent basis to make sure you're accomodating siderial vs synodic days 
        shadowSteps = np.shape(self.shadows)[0]
        
        # Equilibrate 
        print ("-------------Beginning Equilibration---------------")
        i = 0
        vfUpdate = True
        while self.t < self.equiltime or self.nu < np.radians(endTrueAnomaly): #optional input of TA to stop at (works if you're doing complete orbits for equilibration)
            # Update orbit & binary angles 
            self.updateOrbit(equilibration = True)
            # Get slice of shadow and view factor arrays that correspond to current position 
            shadowVals = self.sliceShadows(shadowSteps) # Slice the sections of lookup arrays you need  
            self.advance(shadowVals, vfUpdate)
                
        if endTrueAnomaly != 0.0:       
            print ("Time at end of equilibration: {} s".format(self.t))
            print ("Target True Anomaly/Distance was: {}".format(endTrueAnomaly))
            print ("     Actual TA was: {}".format(np.degrees(self.nu)))
            print ("     Actual r was: {}".format(self.r))
                
        print ("Equilibration reached. Saving temperatures")
        # Run through end of model and store output
        self.dt = self.dtout
        self.t = 0.0  # reset simulation time
         
        for i in range(0, self.N_steps):
            self.updateOrbit()
            shadowVals = self.sliceShadows(shadowSteps)
            self.advance(shadowVals,vfUpdate = True)
            self.T[i,:,:] = self.profile.T # temperature [K]
            self.lt[i] = np.asarray([self.t / self.planet.day * 24.0, self.theta, self.nu])  # local time [hr], theta (rad), true anomaly (rad)
            #vfUpdate = not vfUpdate
            
            # #***
            # self.OutFlux[i] = self.Qs
            # self.OutIndex[i] = self.indices
        print ("VF Update called {} times".format(self.vfCount))
 
    def advance(self,shadowVals,vfUpdate = False):
        # Assuming geometry is handled separately 
        #   Aka view factors, shadowing 
        
        # Finding Qs for each facet 
        reflectedFlux = self.surfModelFlux_Vectorized(shadowVals,self.profile)
        
        if vfUpdate: # Set to calculate view factors at every time step 
               
            # Only do T^4 once since it's a bit slow computationally
            temps4 = np.power(self.profile.T[0,:],4)
            
            # Scattered Flux/reflections 
            # Single body so no interbody 
            self.totalScattered = totalScatteredFluxVectorized2(self.viewFactors, self.profile.Qs, temps4, reflectedFlux, self.planet.albedo)
                  
        # Add scattered flux to Qs
        self.profile.Qs = self.profile.Qs + self.totalScattered

        # Primary
        self.profile.update_T(self.dt, self.profile.Qs, self.planet.Qb)
        self.profile.update_cp()
        self.profile.update_k()
        
        # Increment time
        self.t += self.dt 
        
        
    
    def updateOrbit(self,equilibration = False):
        # New update orbit needs to change orbital values, increase theta and phi 
        orbits.orbitParams(self)
        orbits.updateAngles(self, self.dt, equilibration) # Update theta and phi based on the last time step 
        self.nu += self.nudot * self.dt
        if self.nu >= 2*np.pi:
            self.nu = self.nu - 2*np.pi
        
    def interpolateFluxes(self):
        divisions = len(self.fluxes) 
        hourAngles = np.linspace(0,TWOPI,divisions)
        fluxInterp = interp.interp1d(hourAngles,self.fluxes)
        return fluxInterp
        
    def NormVisCheck(self,tri,ray,local = True):
        # first takes the dot product of solar vector with the facet normal
        # Make negative if doing single (due to flip of axes) 
        if local: 
            rayDirection = -np.asarray(ray.d / np.linalg.norm(ray.d))
        else: 
            rayDirection = np.asarray(ray.d / np.linalg.norm(ray.d))
        dotproduct = np.dot(rayDirection,tri.normal)
        i = np.arccos(dotproduct) # solar incidence angle 
        if i <= (np.pi / 2.0) or i >= ((3*np.pi) / 2.0): #if on day side
            # If less than 90 deg, cast ray to see if visible 
            return True, dotproduct # on day side
        return False, dotproduct # on night side
    

  
    def surfModelFlux_Vectorized(self,shadows, profile):
        # Using shadow arrays, calc flux on each facet
        #     given current orbital position 
        
        # Intensity of sun at current orbital distance times dot products. Accounts for albedo
        insolation = self.solarFlux * shadows
        profile.Qs = (1.0 - self.planet.albedo) * insolation 
        self.Qs = profile.Qs
        
        return self.planet.albedo * insolation
        
            
    def fluxIntensity(self,solarDist,rDist,sVect,normal):
        # Called from FluxCalc
        flux = (solarDist / (rDist**2)) * np.dot(sVect,normal)
        return flux
    
    def findClosestIndexInLookupArrays(self,orientations: np.array):
        # Given angle and the parameters
        #     of a lookup array, find the index of the array that is closest to 
        #     the current orientation
        # ArrayOrientations is an array of the rotational orientations that a 
        #     lookup array covers. It has the same number of indices as the lookup
        #     array has entris 
        index = diff(orientations, np.degrees(self.theta)).argmin()
        
        return index 
    
    def sliceShadows(self,steps):
        # Given the current value of theta, returns the sections of the 
        #     shadowing lookup arrays that most closely corresponds to current 
        #     position (model.theta and model.phi) for primary and secondary 
        
        stepSize = 360. / steps
        orientations = np.arange(0,360,stepSize) # Includes 0, stops before 360
        
        index = self.findClosestIndexInLookupArrays(orientations)
        
        # Slice array
        shadowSlice = self.shadows[index]
        
        return shadowSlice
    
    def sliceViewFactors(self,steps = 70):
        secStep = 360. / steps
        secOrientations = np.arange(steps) * secStep
        
        if self.phi > self.theta:
            separation = self.phi - self.theta + 2*np.pi
        else: 
            separation = self.phi - self.theta
        
        vfIndex = diff(secOrientations, np.degrees(separation)).argmin() 
        return self.binaryVF[vfIndex]
             
    
    
    
    
    
    
    
##############################################################################
## Profile Class
##############################################################################
class profile(object):
    """
    Profiles are objects that contain the model layers
    
    In this implementation, there will be 1 profile with information about
    all facets stored in arrays with at least one dimension the same length
    as the number of facets 
    
    The profile class defines methods for initializing and updating fields
    contained in the model layers, such as temperature and conductivity.
    
    """
    
    #def __init__(self, planet=planets.Moon, facetNum = np.float, lat = np.array, fluxInt = np.array, shadows = np.array,emis = 0.95):
    
    def __init__(self, planet=planets.Moon, facetNum = np.float, lat = np.array,emis = 0.95):

        # planet         
        self.planet = planet
    
        # Number of facets
        self.facetNum = facetNum
                
        # latitude: array?
        self.lat = lat
        
        # emissivity 
        self.emissivity = emis
        

        ######################################################################
        ######################################################################
        
        # to figure out: geometry, lat, long, dec 
        # arrays for facet dependent properties like albedo can go here 
        
        # Initialize surface flux 
        self.Qs = np.zeros(facetNum) #np.float() # surface flux
        
        ######################################################################
        ######################################################################
        
        # The spatial grid
        ks = planet.ks
        kd = planet.kd

        rhos = planet.rhos
        rhod = planet.rhod
        H = planet.H
        cp0 = planet.cp0
        kappa = ks/(rhos*cp0)
        
        #self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b)
        self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b,self.facetNum)
        self.nlayers = np.shape(self.z)[0] # number of model layers (grid is same for all facets, so use [0])        
        self.dz = np.diff(self.z[:,0]) # difference along a given axis. Same across all columns (use 1st col)
        self.d3z = self.dz[1:]*self.dz[0:-1]*(self.dz[1:] + self.dz[0:-1])
        self.g1 = 2*self.dz[1:]/self.d3z[0:] # A.K.A. "p" in the Appendix
        self.g2 = 2*self.dz[0:-1]/self.d3z[0:] # A.K.A. "q" in the Appendix
        
        # Thermophysical properties
        self.kc = kd - (kd-ks)*np.exp(-self.z/H)
        self.rho = rhod - (rhod-rhos)*np.exp(-self.z/H)
        
        # Initialize temperature profile
        self.init_T(planet)
        
        # Initialize conductivity profile
        self.update_k()
        
        # Initialize heat capacity profile
        self.update_cp()
        

        

    
    # Temperature initialization
    def init_T(self, planet=planets.Moon, lat = 0):
        self.T = np.zeros([self.nlayers, self.facetNum]) \
                 + T_eq(planet, lat)#self.lat)
    
    
    # Heat capacity initialization
    def update_cp(self):
        self.cp = heatCapacity(self.planet, self.T)
        #self.cp = heatCapacity_ice(self.T)
    
    
    # Thermal conductivity initialization (temperature-dependent)
    def update_k(self):
        self.k = thermCond(self.kc, self.T)
        
    
    ##########################################################################
    # Core thermal computation                                               #
    # dt -- time step [s]                                                    #
    # Qs -- surface heating rate [W.m-2]                                     #
    # Qb -- bottom heating rate (interior heat flow) [W.m-2]                 #
    ##########################################################################
                         
    def update_T(self, dt, Qs = np.array, Qb = 0):#0, Qb = 0):
        # Coefficients for temperature-derivative terms
        alpha = np.transpose(self.g1*self.k[0:-2].T)
        beta = np.transpose(self.g2*self.k[1:-1].T)
        # Temperature of first layer is determined by energy balance
        # at the surface
        surfTemp(self, Qs)
        # Temperature of the last layer is determined by the interior
        # heat flux
        botTemp(self, Qb)
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        self.T[1:-1,:] = self.T[1:-1,:] + dt/(self.rho[1:-1,:]*self.cp[1:-1,:]) * \
                     (alpha*self.T[0:-2,:] - \
                       (alpha+beta)*self.T[1:-1,:] + \
                       beta*self.T[2:,:] )


     ##########################################################################   
    
    # Simple plot of temperature profile
    def plot(self):
        ax = plt.axes(xlim=(0,400),ylim=(np.min(self.z),np.max(self.z)))
        plt.plot(self.T, self.z)
        ax.set_ylim(1.0,0)
        plt.xlabel('Temperature, $T$ (K)')
        plt.ylabel('Depth, $z$ (m)')
        mpl.rcParams['font.size'] = 14
    
    # Initialize arrays for temperature, lt 
    def defineReadoutArrays(self,N_steps, N_z,facetNum):
        self.readOutT = np.zeros([N_steps, N_z,facetNum])
        self.readOutlT = np.zeros([N_steps])

#---------------------------------------------------------------------------
"""

The functions defined below are used by the thermal code.

"""
#---------------------------------------------------------------------------

# Thermal skin depth [m]
# P = period (e.g., diurnal, seasonal)
# kappa = thermal diffusivity = k/(rho*cp) [m2.s-1]
def skinDepth(P, kappa):
    return np.sqrt(kappa*P/np.pi)


# The spatial grid is non-uniform, with layer thickness increasing downward
def spatialGrid(zs, m, n, b,facetNum):
    # Each column represents a new facet 
    dz = np.zeros([1,facetNum]) + zs/m # thickness of uppermost model layer
    z = np.zeros([1,facetNum]) # initialize depth array at zero
    zmax = zs*b # depth of deepest model layer

    i = 0
    while (np.any(z[i,:]) < zmax):
        i += 1
        h = dz[i-1,:]*(1+1/n) # geometrically increasing thickness
        dz = np.append(dz, [h],axis = 0) # thickness of layer i (axis = 0 --> across rows ie columns)
        z = np.append(z, [z[i-1,:] + dz[i,:]],axis = 0) # depth of layer i (axis = 0 --> across rows ie columns)

    return z


# Solar incidence angle-dependent albedo model
# A0 = albedo at zero solar incidence angle
# a, b = coefficients
# i = solar incidence angle
def albedoVar(A0, a, b, i):
    return A0 + a*(i/(np.pi/4))**3 + b*(i/(np.pi/2))**8

# Radiative equilibrium temperature at local noontime
def T_radeq(planet, lat):
    return ((1-planet.albedo)/(sigma*planet.emissivity) * planet.S * np.cos(lat))**0.25

# Equilibrium mean temperature for rapidly rotating bodies
def T_eq(planet, lat = 0):
    return T_radeq(planet, lat)/np.sqrt(2)

# Heat capacity of regolith (temperature-dependent)
# This polynomial fit is based on data from Ledlow et al. (1992) and
# Hemingway et al. (1981), and is valid for T > ~10 K
# The formula yields *negative* (i.e. non-physical) values for T < 1.3 K
def heatCapacity(planet, T):
    c = planet.cpCoeff
    return np.polyval(c, T)

# Temperature-dependent thermal conductivity
# Based on Mitchell and de Pater (1994) and Vasavada et al. (2012)
def thermCond(kc, T):
    return kc*(1 + R350*T**3)

# Surface temperature calculation using Newton's root-finding method
# p -- profile object
# Qs -- heating rate [W.m-2] (e.g., insolation and infared heating)
    # Array same length as number of facets with Qs for each 
    
def surfTemp(p, Qs):
    Ts = p.T[0,:]
    deltaT = Ts
    
    while (np.any(np.abs(deltaT) > DTSURF)):
        x = p.emissivity*sigma*Ts**3
        y = 0.5*thermCond(p.kc[0,:], Ts)/p.dz[0]
    
        # f is the function whose zeros we seek
        f = x*Ts - Qs - y*(-3*Ts+4*p.T[1,:]-p.T[2,:])
        # fp is the first derivative w.r.t. temperature        
        fp = 4*x - \
             3*p.kc[0,:]*R350*Ts**2 * \
                0.5*(4*p.T[1,:]-3*Ts-p.T[2,:])/p.dz[0] + 3*y
        
        # Estimate of the temperature increment
        deltaT = -f/fp
        Ts += deltaT
    # Update surface temperature
    p.T[0,:] = Ts

# Bottom layer temperature is calculated from the interior heat
# flux and the temperature of the layer above
def botTemp(p, Qb):
    #p.T[-1] = p.T[-2] + (Qb/p.k[-2])*p.dz[-1]
    p.T[-1,:] = p.T[-2,:] + (Qb / p.k[-2,:])*p.dz[-1]

def getTimeStep(p, day):
    dt_min = np.min( F * p.rho[:-1,0] * p.cp[:-1,0] * p.dz**2 / p.k[:-1,0] )
    return dt_min

# Adjusted for highest temperature conductivity 
def getHighTempTimeStep(p, highTk, day):
    dt_min = np.min( F * p.rho[:-1,0] * p.cp[:-1,0] * p.dz**2 / highTk[:-1,0] )
    return dt_min

 
# Used to determine closest index for lookup data
# Takes into account the fact the circular nature of degrees
#    ie. 359 is closer to 0 than to 336 
def diff(a, b, turn=360): 
    return np.minimum((np.remainder(a - b,turn)),np.remainder(b-a,turn))

# Returns the max theoretical temperature at the equator during perihelion
# Used to determine a stable timestep 
def getPerihelionSolarIntensity(planet):
    x = planet.rAU   * (1 - planet.eccentricity**2)
    peDist = x / (1 + planet.eccentricity * np.cos(0)) # Cos 0 corresponds to true anomaly at perihelion
    peIntensity = solarLum / (peDist*1.496e11)**2 #[W.m-2] Intensity of sun at perihelion. Convert AU to m
    peMaxTemp = ((1-planet.albedo)/(sigma*planet.emissivity) * peIntensity * np.cos(0))**0.25 # Max temperature, at equator, at perihelion 
    return peMaxTemp 



#@jit(nopython = True)
def totalScatteredFluxVectorized(viewFactors: np.array, fluxes: np.array, temps: np.array,reflectedFlux: np.array,albedo, emis = 0.95, readOut = False):

    # Amount of flux reflected from each facet (available to other facets as reflection)
    visReflected = reflectedFlux#np.multiply(albedo,fluxes)
    
    q_Vis = np.asarray((1 - albedo) * np.multiply(viewFactors,visReflected[:,np.newaxis])) # Need to take into account how much is absorbed into recieving facet

    # Infrared emission from other facets (absorbed)
    q_IR = emis**2 * sigma * np.multiply(viewFactors,np.power(temps,4))
    
    # Add solar, visible reflected and absorbed scattered infrared light         
    # Axis = 0 sums columns 
    visSum = np.sum(q_Vis, axis=0)
    irSum = np.sum(q_IR, axis = 0)
    fluxTotal = fluxes + visSum + irSum
    

    return fluxTotal

@jit(nopython = True)
def totalScatteredFluxVectorized2(viewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array,albedo, emis = 0.95):    
    # temps4 = np.power(temps,4)
    visReflected = reflectedFlux#np.multiply(albedo,fluxes)
    facets = np.shape(fluxes)[0]
    fluxTotals = np.zeros(facets)
    for i in range(np.shape(fluxes)[0]):
        Vis = np.asarray((1-albedo) * np.multiply(viewFactors[:,i],visReflected))
        # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
        IR = emis**2 * sigma * np.multiply(viewFactors[:,i],temps) 
        visSum = np.sum(Vis)
        irSum = np.sum(IR)
        
        # Output only additional scattered/reflected flux
        fluxTotals[i] = visSum + irSum

    return fluxTotals

@jit(nopython = True)
def interBodyScatteredFlux(binaryViewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array, body, facetNum, albedo, emis = 0.95):
    # These will use the flux/temps of the opposite body (ie primary will use secondary values to calculate scattering)
    fluxTotals = np.zeros(facetNum) # Storage array. Size is number of facets of body you're calculating it for 
    #temps4 = np.power(temps,4)
    for i in range(facetNum):
        if body == "primary":
            Vis = np.asarray((1-albedo) * np.multiply(binaryViewFactors[i,:],reflectedFlux)) # Use rows for primary
            # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
            IR = emis**2 * sigma * np.multiply(binaryViewFactors[i,:],temps)
        elif body == "secondary":
            Vis = np.asarray((1-albedo) * np.multiply(binaryViewFactors[:,i],reflectedFlux)) # Use rows for primary
            # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
            IR = emis**2 * sigma * np.multiply(binaryViewFactors[:,i],temps)
        else: print ("Neither primary or secondary selected for interbody scattering")
        visSum = np.sum(Vis)
        irSum = np.sum(IR)
        
        # Output only additional scattered/reflected flux
        fluxTotals[i] = visSum + irSum
        
    return fluxTotals

@jit(nopython = True)
def singleScatteredMP(viewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array,albedo):
    emis = 0.95
    Vis = np.asarray((1-albedo) * np.multiply(viewFactors,reflectedFlux))
    # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
    IR = emis**2 * sigma * np.multiply(viewFactors,temps) 
    visSum = np.sum(Vis)
    irSum = np.sum(IR)
            
    return visSum + irSum

@jit(nopython = True)
def binaryScatteredMP(binaryViewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array, albedo):
    emis = 0.95
    Vis = np.asarray((1-albedo) * np.multiply(binaryViewFactors,reflectedFlux)) # Use rows for primary
    # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
    IR = emis**2 * sigma * np.multiply(binaryViewFactors,temps)
    visSum = np.sum(Vis)
    irSum = np.sum(IR)
    return visSum + irSum


@jit(nopython = True)
def viewFactorsProcess(index, viewFactors: np.array,binaryViewFactors: np.array, secFluxes: np.array, priFluxes: np.array, \
                       secTemps: np.array, priTemps: np.array, secReflectedFlux: np.array, priReflectedFlux: np.array, priFacetNum, priAlbedo, secAlbedo):
    # Each process will represent one facet so i will be priFacetNum + secFacetNum
    if index < priFacetNum: 
        i = index
        # Individual body
        # Each of these processes will only get pertinent view factors 
        single = singleScatteredMP(viewFactors, priFluxes, priTemps, priReflectedFlux, priAlbedo)
        # Binary
        binary = binaryScatteredMP(binaryViewFactors[i,:], secFluxes, secTemps, secReflectedFlux, priAlbedo)
    else: 
        i = index - priFacetNum
        single = singleScatteredMP(viewFactors, secFluxes, secTemps, secReflectedFlux, secAlbedo)
        binary = binaryScatteredMP(binaryViewFactors[:,i], priFluxes, priTemps, priReflectedFlux, secAlbedo)
        
    return single + binary 



