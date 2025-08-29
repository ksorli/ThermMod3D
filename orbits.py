"""
This module contains functions for calculating solar
angles from orbital elements

Last edited by Kya Sorli on Sept. 2nd, 2021
"""

import numpy as np
import shapeModule as shape

# Constants
AU = 1.49598261e11 # Astronomical Unit [m]
GM = 3.96423e-14 # G*Msun [AU**3/s**2]
TWOPI = 6.283185307
sigma = 5.67051196e-8  # Stefan-Boltzmann Constant (W m^-2 K^-4)
solarLum = (3.826e26 / (4.0*np.pi)) # Solar Luminosity divided by 4pi steradians



def orbitParams(model): # Temp edit to make model.planet --> model.priPlanet
    if hasattr(model, "planet"):
        planet = model.planet
    else:
        planet = model.priPlanet
    a = planet.rAU                # Semi-major axis in AU
    ecc = planet.eccentricity     # Eccentricity
    nu = model.nu                 # True Anomaly (position of a planet on it's orbit; nu = 0 is perihelion, pi radians is aphelion) 
    obliq = planet.obliquity      # Obliquity 
    Lp = planet.Lp                # Longitude of perihelion (Orientation of orbit, angle between perihelion line and arbitrary reference line )
    
    # Useful parameter:
    x = a*(1 - ecc**2)
    
    # Distance to Sun
    model.r = x/(1 + ecc*np.cos(nu))
    
    # Update solar location
    # updateSol(model, model.r)
    
    # Solar declination
    model.dec = np.arcsin( np.sin(obliq)*np.sin(nu+Lp) )
    
    # Angular velocity
    model.nudot = model.r**-2 * np.sqrt(GM*x)
    
    # Solar flux intensity at this distance to sun
    model.solarFlux = solarLum / (model.r*1.496e11)**2 #[W.m-2] Intensity of sun at distance of asteroid. Convert AU to m

    # Solar vector updated with nu 
    model.sVect = updateSolarPos(model.nu)
    
    
    
def interpObliquity(planet, nuInput): # Temp edit to make model.planet --> model.priPlanet
    a = planet.rAU                # Semi-major axis in AU
    ecc = planet.eccentricity     # Eccentricity
    nu = nuInput                # True Anomaly (position of a planet on it's orbit; nu = 0 is perihelion, pi radians is aphelion) 
    obliq = planet.obliquity      # Obliquity 
    Lp = planet.Lp                # Longitude of perihelion (Orientation of orbit, angle between perihelion line and arbitrary reference line )
    
    # Useful parameter:
    x = a*(1 - ecc**2)
    
    # Distance to Sun
    r = x/(1 + ecc*np.cos(nu))
    
    # Angular velocity
    nudot = r**-2 * np.sqrt(GM*x)

    # Solar vector updated with nu 
    sVect = updateSolarPos(nu, dist = r)
    
    return nu, nudot, sVect
    
    
    
def advanceBinaryOrbit(model,priModel: shape, secModel: shape, priTheta,secTheta):
    # Rotates primary about own axis by a given angle (deg.)
    # Rotates secondary about primary by a given angle (deg.)
    
    # Used to setup shadowing and view factors, or if doing shadowing calcs with model 
    # Not called during vectorized version of model (no mesh manipulation there)
    
    # Rotate primary and update vertice/face information 
    priModel.rotateMesh(priTheta)
    
    # Rotates secondary about primary (only for tidally locked secondary)
    secModel.rotateMesh(secTheta)
    
    allTrisNew = []
    allTrisNew.append(priModel.tris)
    allTrisNew.append(secModel.tris)
    
    return allTrisNew

def advanceBinaryOrbit_NoModel(priModel: shape, secModel: shape, priTheta,secTheta):
    # Rotates primary about own axis by a given angle (deg.)
    # Rotates secondary about primary by a given angle (deg.)
    
    # Used to setup shadowing and view factors, or if doing shadowing calcs with model 
    # Not called during vectorized version of model (no mesh manipulation there)
    
    # Rotate primary and update vertice/face information 
    priModel.rotateMesh(priTheta)
    
    # Rotates secondary about primary (only for tidally locked secondary)
    secModel.rotateMesh(secTheta)
    
    allTrisNew = []
    allTrisNew.append(priModel.tris)
    allTrisNew.append(secModel.tris)
    
    return allTrisNew

def updateBinaryAngles(model, timestep,equilibration = False):
    # Using the most recent timestep, update the primary theta and secondary phi
    # Also change the index of shadowing and viewfactor arrays that are closest
    #     This uses resolution from those arrays 
    # Should be called in updateOrbit function 
    
    # Get of movement with time 
    # Primary rotation speed
    thetaDot =  TWOPI / model.priPlanet.day # Radians per second 
    # Secondary orbital speed 
    phiDot = TWOPI / model.secPlanet.day# Radians per second 
    
    # Get angular change 
    thetaChange, phiChange = model.dt * thetaDot, model.dt * phiDot
    if model.t == 0. and equilibration == True:
        print ("Primary rotation per timestep: {} degrees".format(np.degrees(thetaChange)))
        print ("Secondry orbit progression per timestep: {} degrees".format(np.degrees(phiChange)))
        
        # Initiate variable of secondary's location 
        model.lastSecondaryLocation = np.asarray([0,0,0])
    
    # Update 
    model.theta += thetaChange
    model.phi += phiChange
    
    # Needs to be circular (Reset at 2pi radians) 
    if model.theta >= TWOPI: model.theta = model.theta - TWOPI
    if model.phi >= TWOPI: model.phi = model.phi - TWOPI
    
    if not equilibration:
        # Aren't calculating BYORP until output 
        # Mesh location and Secondary Direction Vector (used for forces and BYORP)
        model.secondaryDirectionVector = np.asarray(model.activeSecModel.meshCentroid - model.lastSecondaryLocation)
        model.lastSecondaryLocation = model.activeSecModel.meshCentroid
    
def updateAngles(model, timestep,equilibration = False):
    # Called in UpdateOrbit function for single body model 
    # Get of movement with time 
    # Rotation speed 
    thetaDot =  TWOPI / model.planet.day # Radians per second 
    
    # Get angular change 
    thetaChange = model.dt * thetaDot
    if model.t == 0. and equilibration == True:
        print ("Rotation per timestep: {} degrees".format(np.degrees(thetaChange)))
    
    # Update 
    model.theta += thetaChange
    
    # Needs to be circular (Reset at 2pi radians) 
    if model.theta >= TWOPI: model.theta = model.theta - TWOPI




# NEXT TWO FUNCTIONS SHOULD PROBABLY BE MOVED TO MAIN FILE 
def diff(a, b, turn=360): 
    # Used to determine closest lookup data
    # Takes into account the fact the circular nature of degrees
    #    ie. 359 is closer to 0 than to 336 
    return np.minimum((np.remainder(a - b,turn)),np.remainder(b-a,turn))


def findClosestIndexInLookupArrays(model,priOrientations: np.array, secOrientations: np.array):
    # Given two angles (primary theta and secondary phi) and the parameters
    #     of a lookup array, find the index of the array that is closest to 
    #     the current orientation
    # ArrayOrientations is an array of the rotational orientations that a 
    #     lookup array covers. It has the same number of indices as the lookup
    #     array has entris 
    
    priIndex = diff(priOrientations, model.theta).argmin()
    secIndex = diff(secOrientations, model.phi).argmin()
    
    return priIndex, secIndex 
    
    
    
    
    
def cosSolarZenith(lat, dec, h):
    
    # Cosine of solar zenith angle
    x = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(h)
    
    # Clipping function = zero when sun below horizon:
    y = 0.5*(x + np.abs(x)) 
    
    return y

def updateSol(model,theta = 0.):
    model.sLoc = np.asarray([0,0,model.r])
    
    sLoc = model.sLoc
    if theta != 0.:
        model.newSLoc = np.asarray(shape.rotate(sLoc,model.theta)) 
    #Currecntly sun is located on the x axis. Eventually, incorporate dec/other metrics
        
def updateSolarPos(nu,sunStartVec = np.asarray([1,0,0]),sunStartPoint = np.asarray([-1,0,0]), dist = None):
    # Update the position of the sun 
    # Sun is currently moving around the body 
    
    # Use true anomaly to move vector around body 
    # nu is an angle. 0 is perihelion and pi radians is aphelion 
    # Start with initial point/vector at perihelion 
    if dist == None:
        startVec = sunStartVec
    else: 
        # If doing raytracing, need to know how far away sun is 
        startVec = np.asarray([dist*1.496e11,0,0])
    rotz = np.asarray([np.cos(nu),-np.sin(nu),0,np.sin(nu),np.cos(nu),0,0,0,1]).reshape((3,3))
    newVec = np.matmul(rotz, startVec)
    
    # startPoint = sunStartPoint
    # newPoint = np.matmul(rotz,startPoint)
    
    # Point
    # newPoint = np.matmul(rotz,sunStartPoint)
    # Need to adjust distance based on model.r
    return newVec

def hourAngle(t, P):
    
    return (TWOPI * t/P) % TWOPI

def hourAngle_Second(t, P):
    
    return (TWOPI * t/P) % TWOPI


    