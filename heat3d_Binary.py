#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:48:00 2023

@author: kyso3185
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:51:23 2022

@author: kyso3185

Former iteration was heat3d_shadowing.py. That did shadowing but used multistepping
approach. Since facets need to communicate, this builds on that by vectorizing 
the actual model process and allowing numpy to do the work. 

This iteration of the model can take in precomputed shadows and view factors
for a full binary pair, and calculates temperatures for both bodies. View factors
can be set to off by using "None" in the place of the file path to precomputed 
view factor values.

The model finds the closest value in the lookup arrays based on the angular 
location of both the primary and the secondary. After equilibration completes, 
the user has the option of selecting "AdaptiveShadows", which will perform 
ray tracing on the fly for each timestep. This is more computationally intensive 
but gives exact shadows for each location. This is suggested for any model run
including BYORP calculations. 

"""


# Physical constants:
sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
chi = 2.7 # Radiative conductivity parameter [Mitchell and de Pater, 1994]
R350 = chi/350**3 # Useful form of the radiative conductivity
TWOPI = 6.283185307

# Numerical parameters:
F = 0.5 # Fourier Mesh Number, must be <= 0.5 for stability
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
#import math

# MatPlotLib and Pyplot are used for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Methods for calculating solar angles from orbits
import orbits

# Planets database
import planets

# Read in Shape Model
#import shapeModel as shape
import shapeModule as shape


# For interpolation functions
import scipy.interpolate as interp

from numba import jit

# Copy
import copy 

# Shadows 
import shadowModule


# BYORP
import BYORPModule as BYORP


#Set up solar values and physical constants 
sigma = 5.67051196e-8  # Stefan-Boltzmann Constant (W m^-2 K^-4)
solarLum = (3.826e26 / (4.0*np.pi))
solar = 3.826e26 / (4.0*np.pi)
s0 = solarLum  #[W.m-2] Intensity of sun at distance of asteroid 





##############################################################################
## Binary Model Class 
##############################################################################
class binaryModel(object):
    def __init__(self, priShapeModelPath, secShapeModelPath, shapeModelRotation = 0, priPlanet = planets.Moon,secPlanet = planets.Moon, ndays=1,\
                 nyears = 1.0,local = False,priShadowLookupPath = None, secShadowLookupPath = None,priVF = None,secVF = None,\
                     binaryVF = None, flippedBinaryVF = None, priHighResShadowPath = None, secHighResShadowPath = None, \
                         priHighResOrient = None, secHighResOrient = None, noBVF = False, noEclipses = False, \
                             secondaryShift = (2.46,0,0), AdaptiveShadows = False,shift = None, calcBYORP = False):
    
        # Change Equilibration Time
        NYEARSEQ = nyears   
        print ("Equilibration time: {} years".format(NYEARSEQ))
    
        # Initialize
        self.priPlanet = priPlanet
        self.secPlanet = secPlanet
        self.Sabs = self.priPlanet.S * (1.0 - self.priPlanet.albedo)
        self.r = self.priPlanet.rAU # solar distance [AU]
        self.nu = np.float() # orbital true anomaly [rad]
        self.nudot = np.float() # rate of change of true anomaly [rad/s]
        self.dec = np.float() # solar declination [rad]
        self.solarFlux = np.float() # Solar flux (W/m^2) at the body given the distance to the sun
       
        # Read in Shape Model & initialize corresponding values
        self.local = local
        self.AdaptiveShadows = AdaptiveShadows 
        if AdaptiveShadows:
            print ("Adaptive shadows True. On the fly ray tracing will be used")
            # Add shift and rotation to secondary for on the fly ray tracing 
            # Rotates secondary shape model to get correct orientation 
            # secRot should be 90 if shape model long axis is natively parallel to direction of motion and 0 if not 
            print ("Primary Shape Parameters?")
            self.priShape = shape.shapeModel(priShapeModelPath,local)
            print ("Secondary Shape Parameters?")
            self.secShape = shape.shapeModel(filePath = secShapeModelPath,shapeModelRotation = shapeModelRotation, local = local, shift = secondaryShift)

        else:
            print ("Primary Shape Parameters?")
            self.priShape = shape.shapeModel(priShapeModelPath,local)
            print ("Secondary Shape Parameters?")
            self.secShape = shape.shapeModel(secShapeModelPath,local, shift = secondaryShift)
        self.priFacetNum = self.priShape.facetNum
        self.secFacetNum = self.secShape.facetNum
        self.facetNum = self.priFacetNum + self.secFacetNum
        print ("Shape Model Loaded")
        
        # Initialize arrays for latitudes, fluxes, shadows 
        self.allLats = np.concatenate([self.priShape.lats,self.secShape.lats]) # np array of latitudes for each facet for both bodies
        
        # Rotation of the primary about own axis 
        # Used for shadowing and view factor setup  
        self.theta = np.float() # Radians, starts at 0.
        
        # Rotation of the secondary about primary
        # Used for shadowing setup 
        self.phi = np.float() # Radians, starts at 0
        
        # Load shadow arrays 
        self.secShadows = np.load(secShadowLookupPath)
        self.priShadows = np.load(priShadowLookupPath)
        self.noEclipses = noEclipses # Indicate if shadows are for independent, single bodies (no eclipses or moonshine) 
        
        # High resolution shadows (no longer in use now that on the fly ray tracing is implemented. Future versions will remove this entirely)
        if priHighResShadowPath != None:
            print ("High resolution shadows provided")
            self.highResShadows = True
            self.priHighResShadows = np.load(priHighResShadowPath)
            self.secHighResShadows = np.load(secHighResShadowPath)
            self.priHROrient = np.load(priHighResOrient)
            self.secHROrient = np.load(secHighResOrient)
            
        else: 
            self.highResShadows = False
            

        # Determine the sections of orbit where eclipses are geometrically possible 
        maxBinarySeparation = self.priShape.maxDim + self.secShape.maxDim
        print ("Maximum primary-secondary separation: {} km".format(maxBinarySeparation))
        vecPriSec = np.linalg.norm(np.asarray(self.secShape.meshCentroid - self.priShape.meshCentroid))
        maxAngleSep = np.arcsin(maxBinarySeparation / vecPriSec)
        endHRPoint = np.pi * 2 - maxAngleSep
        self.eclipseRange = np.asarray([maxAngleSep, endHRPoint])
        print ("Shadow arrays loaded")
        
        
        # Initialize profile(s)        
        # Two profiles (one for each body)
        print ("Primary-----------")
        self.priProfile = profile(planet = self.priPlanet, facetNum = self.priFacetNum, lat = self.priShape.lats, emis = self.priPlanet.emissivity)
        print ("Secondary---------")
        self.secProfile = profile(planet = self.secPlanet, facetNum = self.secFacetNum, lat = self.secShape.lats, emis = self.secPlanet.emissivity)
        print ("Profiles Initialized")
                        
        # Binary model run times.
        # Equilibration time 
        # **Most of this will be done on basis of secondary day. Primary will spin multiple times during this period 
        self.equiltime = NYEARSEQ * secPlanet.year - (NYEARSEQ*secPlanet.year)%secPlanet.day
        # Run time for output
        self.endtime = self.equiltime + ndays*secPlanet.day
        self.t = 0.
        
        # Find min of two timesteps for primary and secondary profiles
        priTimeStep = getTimeStep(self.priProfile, self.priPlanet.day)
        secTimeStep = getTimeStep(self.secProfile, self.secPlanet.day)
        self.dt = min([priTimeStep,secTimeStep])
 
        # Check for maximum time step
        self.dtout = self.dt
        
        print ("timesteps to equiltime: {}".format(self.equiltime / self.dt))
        # Max possible timestep 
        dtmax = secPlanet.day/NPERDAY
        if self.dt > dtmax:
            self.dtout = dtmax
        
        # Array for output temperatures and local times
        # Done on the basis of secondary day in length but using primary's timestep...will be huge readout arrays 
        N_steps = np.int((ndays*secPlanet.day) / self.dtout)
        print ("Nsteps: {}".format(N_steps))
        self.N_steps = N_steps
        
        
        # Binary profile
        if np.shape(self.priProfile.z)[0] != np.shape(self.secProfile.z)[0]:
            print ("Depth layers for two bodies not equivalent. Smaller value recommended")
        N_z = np.shape(self.priProfile.z)[0]
        self.N_z = N_z
        
        # Temperature and local time arrays 
        self.priT = np.zeros([N_steps, N_z,self.priFacetNum])
        self.secT = np.zeros([N_steps, N_z,self.secFacetNum])
        self.lt = np.zeros((N_steps,5)) #[N_steps, 3]) # Time, theta, phi, r, flux
        
        # Scattering readout arrays 
        self.priReadoutScat = np.zeros([N_steps, 2, self.priFacetNum])
        self.secReadoutScat = np.zeros([N_steps, 2, self.secFacetNum])

        self.PriOutIndex = np.zeros((N_steps,2))
        self.PriOutFlux = np.zeros((N_steps,self.priFacetNum))

        # Resolution for view factors and shadowing 
        # NOTE: this is not complete and will be incorporated in future versions of the model  
        if priShadowLookupPath == None or secShadowLookupPath == None:
            print ("Calculating facet resolution around primary equator")
            lats = np.zeros(self.priFacetNum)
            longs = np.zeros(self.priFacetNum)
            equatorial_facets = 0
            for tri in self.priShape.tris:
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
        #* Inidividual bodies 
        if priVF != None and secVF != None:
            self.priVF = np.load(priVF) # Primary
            self.secVF = np.load(secVF) # Secondary 
            self.IncludeReflections = True
            
        else: 
            print ("WARNING: AUTOMATIC VIEW FACTORS NOT ENABLED! \n Proceeding with null view factors")
            self.IncludeReflections = False
            self.priVF = np.zeros((self.priFacetNum,self.priFacetNum))
            self.secVF = np.zeros((self.secFacetNum,self.secFacetNum))
            
        # Binary View factors: inter body
        if binaryVF != None and flippedBinaryVF != None:
            self.binaryVF = np.load(binaryVF)
            #****
            self.flippedBinaryVF = np.load(flippedBinaryVF)
            #****
        else: print ("NO BINARY VIEW FACTORS PROVIDED")
        self.noBVF = noBVF
        if noBVF: 
            print ("NoBVF enabled: binary view factors will not be used")
            self.primaryBinaryScatteredFlux = np.zeros(self.priFacetNum)
            self.secondaryBinaryScatteredFlux = np.zeros(self.secFacetNum)
        else:
            print ("Binary view factors will be used")            
        
                
        # Sun location initialization 
        self.sLoc = np.asarray([0,0,1.496e11]) # 1 AU # This will be scaled to the right distance after first orbital update called 
        self.newSLoc = self.sLoc
        self.baseDist = np.float()
        
        
        # Scattering arrays
        self.priTotalScattered = np.zeros(self.priFacetNum)
        self.secTotalScattered = np.zeros(self.secFacetNum)
        
        print ("Completed Visibility Setup")
        
        
        
        # BYORP and forces
        self.calcBYORP = calcBYORP
        if calcBYORP: 
            print ("calcBYORP is True: Forces and accelerations will be calculated for output time period")
            
            self.netForceVectorStorage = []
            self.accelBYORPStorage = []
            self.byorpRotationVals = [] # theta value at each timestep 
            self.byorpCoeff = None
        
        
        
    def run(self,calcShadows = False, vfFreq = 1, vfOff = False, rayTraceDivisions = 3000):
        # For now assuming precomputed shadows durign equilibration. This will be updated in future versions 
                
        # If using no binary interaction (aka no eclipses or moonshine, shadow arrays have different shapes )
        if not self.noEclipses:
            priSteps = np.shape(self.priShadows)[0]
            secSteps = np.shape(self.priShadows)[1]
        else: 
            priSteps = np.shape(self.priShadows)[0]
            secSteps = np.shape(self.secShadows)[0]
            
        if not self.noBVF:
            vfSteps = np.shape(self.binaryVF)[0]

        # Equilibrate 
        print ("-------------Beginning Equilibration---------------")
        #------------------ Equilibration Start ---------------------------#
        i = 0
        vfUpdate = True
        vfCount = 0
        while self.t < self.equiltime:

            # Update orbit & binary angles 
            self.updateOrbit(equilibration = True)
            
            # If localized high res shadows are being used 
            if self.highResShadows: 
                # If within eclipse territory
                if self.phi <= self.eclipseRange[0] or self.phi >= self.eclipseRange[1]:
                    # Use the high resolution shadows
                    priShadowVals, secShadowVals = self.sliceShadowsWithHighRes(self.priHROrient,self.secHROrient)
                
                else: priShadowVals, secShadowVals = self.sliceShadows(priSteps,secSteps)
            else: priShadowVals, secShadowVals = self.sliceShadows(priSteps,secSteps)   

            if not self.noBVF:
                # Slice view factors for current position 
                binaryVFSlice, flippedBinaryVFSlice = self.sliceViewFactors(vfSteps)
            else: 
                binaryVFSlice, flippedBinaryVFSlice = None, None
            
            # Determine if it's time to update the view factors 
            if vfCount == 0:
                vfUpdate = True
            else: vfUpdate = False
            if vfOff:
                vfUpdate = False
                

            self.advance(priShadowVals, secShadowVals,binaryVFSlice,flippedBinaryVFSlice, vfUpdate)
            i += 1
            vfCount += 1
            
            if vfCount > vfFreq:
                vfCount = 0
          
            
        #------------------ Equilibration Complete ---------------------------#
        print ("Equilibration reached. Saving temperatures")
        # Run through end of model and store output
        self.dt = self.dtout
        self.t = 0.0  # reset simulation time
        
        
        # Setup for OTF ray tracing 
        if self.AdaptiveShadows:
            print ("Adaptive Shadows True: Output will use on the fly ray tracing")
            # Used if doing on the fly raytracing 
            divisions = rayTraceDivisions # How many times you want to do on the fly raytracing in an orbit
            raytraceUpdate = int(self.N_steps / divisions) # Distance between OTF raytraces 
            raytraceCount = 0 # Start at 0 # Keeping count since last raytrace 
            testOTFCount = 0
            
        # Setup for BYORP 
        if self.calcBYORP:
            print ("Radiative forces and accelerations will be calculated")
            # Used if doing on the fly raytracing 
            divisions = 360 # How many times you want to do BYORP calculation in an orbit
    
        # Step through output time 
        for i in range(0, self.N_steps):
            # During output period, orbit update will also update shape models 
            self.updateOrbit()
            
            ###############################################################
            ## Shadowing 
            ###############################################################
            # If within eclipse territory
            if self.phi <= self.eclipseRange[0] or self.phi >= self.eclipseRange[1]:
            
                # If using on the fly ray tracing 
                if self.AdaptiveShadows:
                    # Do raytracing 
                    # Need to import shadow module (currently shadowTestForAtlas would be that)
                    # Could potentially add function directly too but might get messy 
                    # For now, try importing shadowTestForAtlas but realize it might try to execute the rest of the script 
                    if raytraceCount == 0: # Time to do a new ray tracing calculation  
                        
                        # If feeding in rotated shape mdoels 
                        priShadowVals, secShadowVals = shadowModule.binarySingleOrientationShadows(self.activePriModel, self.activeSecModel, 0, 0)
                        testOTFCount += 1
                    
                    # Add an optional tag with how many steps before you want to do this again 
                    raytraceCount += 1
                    if raytraceCount >= raytraceUpdate: # Check if time for new ray tracing calculation 
                        raytraceCount = 0 
                
                # If high res is an option
                elif self.highResShadows: 
                    # High res shadows 
                    priShadowVals, secShadowVals = self.sliceShadowsWithHighRes(self.priHROrient,self.secHROrient)
                    
                
                # Else use normal pre calculated shadows 
                else: priShadowVals, secShadowVals = self.sliceShadows(priSteps,secSteps)
            else: priShadowVals, secShadowVals = self.sliceShadows(priSteps,secSteps)   

            
            ###############################################################
            ## Binary View Factors ****  
            ###############################################################
            if not self.noBVF:
                # Slice view factors for current position 
                binaryVFSlice, flippedBinaryVFSlice = self.sliceViewFactors(vfSteps)
            else: 
                binaryVFSlice, flippedBinaryVFSlice = None, None


            if vfOff:
                vfUpdate = False
            self.advance(priShadowVals, secShadowVals,binaryVFSlice,flippedBinaryVFSlice,vfUpdate)
            
            self.priReadoutScat[i] = np.asarray([self.primaryBinaryScatteredFlux, self.priProfile.Qs])
            self.secReadoutScat[i] = np.asarray([self.secondaryBinaryScatteredFlux, self.secProfile.Qs])
            
            # *** Output T arrays with two profiles 
            self.priT[i,:,:] = self.priProfile.T # primary temperature [K]
            self.secT[i,:,:] = self.secProfile.T # secondary temperature [K]
            self.lt[i] = np.asarray([self.t / self.secPlanet.day * 24.0, self.theta, self.phi, self.r, self.solarFlux])  # local time [hr], theta (rad), phi (rad) 
            
            #***
            self.PriOutFlux[i] = self.priQs
            
            # Update vfUpdate 
            vfUpdate = not vfUpdate # Do view factors every 2 

        
            ###############################################################
            ## BYORP 
            ###############################################################
        
            if self.calcBYORP:
                # If doing it every time
                netForceVector = BYORP.RadiationForces(self, self.secShape, includeAbsorption=False) #bodyFixedFrame = False
                
                
                self.netForceVectorStorage.append(netForceVector)
                self.byorpRotationVals.append(self.phi)
                #self.accelBYORPStorage.append(accelBYORP)
                
                
                # # If doing it only a set number of times in an orbit 
                # if BYORPCount == 0: # Time to do a new ray tracing calculation  
                #     netForceVector, accelBYORP = BYORP.RadiationForces(self)
                #     BYORPExecutionCount += 1


                # # Add an optional tag with how many steps before you want to do this again 
                # BYORPCount += 1
                # if BYORPCount >= BYORPUpdate: # Check if time for new ray tracing calculation 
                #     BYORPCount = 0 

        if self.AdaptiveShadows:
            print ("OTF ray tracing performed: {} times".format(testOTFCount))
        
        
        # BYORP
        # Storage arrays for the BYORP forces and accelerations 
        if self.calcBYORP:
            self.netForceVectorStorage = np.asarray(self.netForceVectorStorage)
            self.byorpRotationVals = np.asarray(self.byorpRotationVals)

            # Normalizing
            normalizedCoefficient = BYORP.NormalizeCoefficients(self,self.netForceVectorStorage,self.byorpRotationVals)
            print ("-------------------------------------------------------")
            print ("-------------------------------------------------------")
            print ("B (A_02 Normalized) Coefficient: {}".format(normalizedCoefficient[1]))
            print ("-------------------------------------------------------")
            print ("-------------------------------------------------------")
            self.byorpCoeff = normalizedCoefficient
        
        
        


            


    def advance(self,priShadowVals, secShadowVals,  binaryVFSlice = None, flippedBinaryVFSlice = None, vfUpdate = True):
        #****
        # Non multiprocessed version of advance that uses one profile object and adds a dimension to each array 
        
        # Assuming geometry is handled separately 
        #   Aka view factors, shadowing 
        
        # Finding Qs for each facet 
        #    Needs to be vectorized
        #    Feed in boolean array (or 0s and 1s) for shadowing state of each facet 
        #****
        priReflectedFlux, secReflectedFlux = self.surfModelFlux_Vectorized_Binary(priShadowVals,secShadowVals,self.priProfile,self.secProfile)
        
        # Reflected fluxes (used for BYORP)
        self.priProfile.reflectedQs = priReflectedFlux
        self.secProfile.reflectedQs = secReflectedFlux
        
        if self.IncludeReflections:
            if vfUpdate:
                # Only do T^4 once since it's a bit slow computationally
                priTemps4 = np.power(self.priProfile.T[0,:],4)
                secTemps4 = np.power(self.secProfile.T[0,:],4)
                
                
                # Single Body Scattered Flux Via View Factors
                primaryScatteredFlux = totalScatteredFluxVectorized2(self.priVF, self.priProfile.Qs, priTemps4, priReflectedFlux, self.priPlanet.albedo)
                secondaryScatteredFlux = totalScatteredFluxVectorized2(self.secVF, self.secProfile.Qs, secTemps4, secReflectedFlux, self.secPlanet.albedo)
                
                # If binary view factors enabled 
                if not self.noBVF: 
                    # Moonshine recieved by primary
                    self.primaryBinaryScatteredFlux = interBodyScatteredFlux(binaryVFSlice, flippedBinaryVFSlice, self.secProfile.Qs, secTemps4, secReflectedFlux, "primary", self.priProfile.facetNum, self.priPlanet.albedo, self.secPlanet.emissivity)
                    # Moonshine recieved by secondary 
                    self.secondaryBinaryScatteredFlux = interBodyScatteredFlux(binaryVFSlice, flippedBinaryVFSlice, self.priProfile.Qs, priTemps4, priReflectedFlux, "secondary", self.secProfile.facetNum, self.secPlanet.albedo, self.priPlanet.emissivity)
                
                # Moonshine and individual
                self.priTotalScattered = primaryScatteredFlux + self.primaryBinaryScatteredFlux
                self.secTotalScattered = secondaryScatteredFlux + self.secondaryBinaryScatteredFlux
                
                self.savePrimaryScat = self.primaryBinaryScatteredFlux
                  
            # Add scattered flux (if not doing view factor calculations each time, this will allow for last time step's values to be maintained)
            self.priProfile.Qs = self.priProfile.Qs + self.priTotalScattered
            self.secProfile.Qs = self.secProfile.Qs + self.secTotalScattered
            
        # Primary
        self.priProfile.update_T(self.dt, self.priProfile.Qs, self.priPlanet.Qb)
        self.priProfile.update_cp()
        self.priProfile.update_k()
        
        # Secondary
        self.secProfile.update_T(self.dt, self.secProfile.Qs, self.secPlanet.Qb)
        self.secProfile.update_cp()
        self.secProfile.update_k()
        
        self.t += self.dt # increment time 
        
        
    
    def updateOrbit(self,equilibration = False):
        # New update orbit needs to change orbital values, increase theta and phi 
        orbits.orbitParams(self)
        
        if not equilibration:
            # Update shape model for BYORP and on the fly ray tracing  
            # Make deep copies of shapes that are actively updating location 
            self.activePriModel = copy.deepcopy(self.priShape)
            self.activeSecModel = copy.deepcopy(self.secShape)
            
            self.activePriModel.rotateMesh(np.degrees(self.theta))
            self.activeSecModel.rotateMesh(np.degrees(self.phi))
            
        
        orbits.updateBinaryAngles(self, self.dt,equilibration) # Update theta and phi based on the last time step 
        self.nu += self.nudot * self.dt
        if self.nu >= 2*np.pi:
            self.nu = self.nu - 2*np.pi
        
    def interpolateFluxes(self):
        divisions = len(self.fluxes) 
        hourAngles = np.linspace(0,TWOPI,divisions)
        fluxInterp = interp.interp1d(hourAngles,self.fluxes)
        return fluxInterp
        
    def NormVisCheck(self,tri,ray,local = True):
        # First takes the dot product of solar vector with the facet normal
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
    
        
    def surfModelFlux_Vectorized_Binary(self,priShadows, secShadows, priProfile, secProfile):
        # Using shadow and view factor (eventually) arrays, calc flux on each facet
        #     given current orbital position 
        # Explicitly requires two profiles 
        
        # Two Profiles 
        # Intensity of sun at current orbital distance times dot products. Accounts for albedo
        priInsolation = self.solarFlux * priShadows
        secInsolation = self.solarFlux * secShadows
        priProfile.Qs = (1.0 - self.priPlanet.albedo) * priInsolation 
        secProfile.Qs = (1.0 - self.secPlanet.albedo) * secInsolation
        
        # ***
        self.priQs = priProfile.Qs
        
        return self.priPlanet.albedo * priInsolation, self.secPlanet.albedo * secInsolation
        

            
    def fluxIntensity(self,solarDist,rDist,sVect,normal):
        # Called from FluxCalc
        flux = (solarDist / (rDist**2)) * np.dot(sVect,normal)
        return flux
    
    def findClosestIndexInLookupArrays(self,priOrientations: np.array, secOrientations: np.array):
        # Given two angles (primary theta and secondary phi) and the parameters
        #     of a lookup array, find the index of the array that is closest to 
        #     the current orientation
        # ArrayOrientations is an array of the rotational orientations that a 
        #     lookup array covers. It has the same number of indices as the lookup
        #     array has entris 
        priIndex = diff(priOrientations, np.degrees(self.theta)).argmin()
        secIndex = diff(secOrientations, np.degrees(self.phi)).argmin()
        
        return priIndex, secIndex 
    
    def sliceShadows(self,priSteps,secSteps):
        # Given the current values of theta and phi, returns the sections of the 
        #     shadowing lookup arrays that most closely corresponds to current 
        #     position (model.theta and model.phi) for primary and secondary 
        #if priSteps != secSteps: print ("Primary and secondary shadowing lookup arrays have different number of orientations")
        
        priStepSize = 360. / priSteps
        secStepSize = 360. / secSteps
        priOrientations = np.arange(0,360,priStepSize) # Includes 0, stops before 360
        secOrientations = np.arange(0,360,secStepSize)
        
        priIndex, secIndex = self.findClosestIndexInLookupArrays(priOrientations,secOrientations)
        # ***
        self.indices = np.asarray([priIndex,secIndex])
        
        if self.noEclipses:
            # If you are not doing interbody radiation exchange (ie moonshine or eclipses)
            # This should only be called when using single body shadows
            # Added for BYORP validation 
            priSlice = self.priShadows[priIndex]
            secSlice = self.secShadows[secIndex]
        else:
            # Pull shadows from full ray trace results 
            priSlice = self.priShadows[priIndex][secIndex]
            secSlice = self.secShadows[priIndex][secIndex]
        
        # print ("Slice: {}".format(priSlice[:10]))
        # print ("Slice size: {}".format(np.shape(priSlice)))
        return priSlice, secSlice
    
    #****
    def sliceShadowsWithHighRes(self,priOrientations,secOrientations):
        # If higher res is done for the eclipse region, this function is used 
        #     Called if phi is within range for possible eclipse 
        # Given the current values of theta and phi, returns the sections of the 
        #     shadowing lookup arrays that most closely corresponds to current 
        #     position (model.theta and model.phi) for primary and secondary 
        
        # priStepSize = 360. / priSteps
        # secStepSize = 360. / secSteps
        # priOrientations = np.arange(0,360,priStepSize) # Includes 0, stops before 360
        # secOrientations = np.arange(0,360,secStepSize)
        
        priIndex, secIndex = self.findClosestIndexInLookupArrays(priOrientations,secOrientations)
        self.indices = np.asarray([priIndex,secIndex])

        # priSlice = self.priShadows[priIndex][secIndex]
        # secSlice = self.secShadows[priIndex][secIndex]
        
        priSlice = self.priHighResShadows[priIndex][secIndex]
        secSlice = self.secHighResShadows[priIndex][secIndex]
        
        # print ("Slice: {}".format(priSlice[:10]))
        # print ("Slice size: {}".format(np.shape(priSlice)))
        return priSlice, secSlice
    #****
    

    def sliceViewFactors(self,steps = 70):
        secStep = 360. / steps
        secOrientations = np.arange(steps) * secStep
        
        if self.phi > self.theta:
            separation = self.phi - self.theta + 2*np.pi
        else: 
            separation = self.phi - self.theta
        
        vfIndex = diff(secOrientations, np.degrees(separation)).argmin()
        #return vfIndex 
        #return self.binaryVF[vfIndex], np.transpose(self.flippedBinaryVF[vfIndex])
        return self.binaryVF[vfIndex], self.flippedBinaryVF[vfIndex]
        #****
       

            
    
    
    
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
        print ("kappa: {}".format(kappa))
        print ("Skin depth: {}".format(np.sqrt(kappa*planet.day/np.pi)))

        self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b,self.facetNum)
        self.nlayers = np.shape(self.z)[0] # number of model layers (grid is same for all facets, so use [0])  
        print ("nlayers: {}".format(self.nlayers))
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
        #alpha = self.g1*self.k[0:-2]
        #beta = self.g2*self.k[1:-1]
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
# Vectorized version 
def spatialGrid(zs, m, n, b,facetNum):
    # Each column represents a new facet 
    dz = np.zeros([1,facetNum]) + zs/m # thickness of uppermost model layer
    z = np.zeros([1,facetNum]) # initialize depth array at zero
    zmax = zs*b # depth of deepest model layer

    i = 0
    while (np.any(z[i,:] < zmax)):
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
def T_eq(planet, lat):
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

# Returns the max theoretical temperature at the equator during perihelion
# Used to determine a stable timestep 
def getPerihelionSolarIntensity(planet):
    x = planet.rAU   * (1 - planet.eccentricity**2)
    peDist = x / (1 + planet.eccentricity * np.cos(0)) # Cos 0 corresponds to true anomaly at perihelion
    peIntensity = solarLum / (peDist*1.496e11)**2 #[W.m-2] Intensity of sun at perihelion. Convert AU to m
    peMaxTemp = ((1-planet.albedo)/(sigma*planet.emissivity) * peIntensity * np.cos(0))**0.25 # Max temperature, at equator, at perihelion 
    return peMaxTemp 

# Adjusted for highest temperature conductivity 
def getHighTempTimeStep(p, highTk, day):
    dt_min = np.min( F * p.rho[:-1,0] * p.cp[:-1,0] * p.dz**2 / highTk[:-1,0] )
    return dt_min

 
# Used to determine closest index for lookup data
# Takes into account the fact the circular nature of degrees
#    ie. 359 is closer to 0 than to 336 
def diff(a, b, turn=360): 
    return np.minimum((np.remainder(a - b,turn)),np.remainder(b-a,turn))


def totalScatteredFluxVectorized(viewFactors: np.array, fluxes: np.array, temps: np.array,reflectedFlux: np.array,albedo, emis = 0.95, readOut = False):
    
    # Amount of flux reflected from each facet (available to other facets as reflection)
    visReflected = reflectedFlux#np.multiply(albedo,fluxes)
    
    q_Vis = np.asarray((1 - albedo) * np.multiply(viewFactors,visReflected[:,np.newaxis])) # Need to take into account how much is absorbed into recieving facet

    # Infrared emission from other facets (absorbed)
    q_IR = emis**2 * sigma * np.multiply(viewFactors,np.power(temps,4))
    
    # Add solar, visible reflected and absorbed scattered infrared light         
    visSum = np.sum(q_Vis, axis=0)
    irSum = np.sum(q_IR, axis = 0)
    fluxTotal = fluxes + visSum + irSum

    return fluxTotal

@jit(nopython = True)
def totalScatteredFluxVectorized2(viewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array,albedo, emis = 0.95):    
    visReflected = reflectedFlux
    facets = np.shape(fluxes)[0]
    fluxTotals = np.zeros(facets)
    #temps4 = np.power(temps, 4)
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
def interBodyScatteredFlux(binaryViewFactors: np.array, flippedViewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array, body, facetNum, albedo, emis = 0.95):
    # These will use the flux/temps of the opposite body (ie primary will use secondary values to calculate scattering)
    fluxTotals = np.zeros(facetNum) # Storage array. Size is number of facets of body you're calculating it for 
    for i in range(facetNum):
        if body == "primary":
            #Vis = np.asarray((1-albedo) * np.multiply(binaryViewFactors[i,:],reflectedFlux)) # Use rows for primary
            Vis = np.asarray((1-albedo) * np.multiply(flippedViewFactors[i,:],reflectedFlux)) # Use rows for primary
            # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
            IR = emis**2 * sigma * np.multiply(flippedViewFactors[i,:],temps)
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

