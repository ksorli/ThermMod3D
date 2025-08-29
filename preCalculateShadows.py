#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:14:40 2023

@author: kyso3185

Module for calculating shadows through ray tracing. Can be applied to 
binary asteroids, single bodies or landforms with a DEM.  

Generates shadowing arrays (.npy)  
"""


import numpy as np
import rayTracing as raytracing 
import kdTree3d as kdtree
import shapeModule as shape
import shadowModule as shadowMod

# Useful
import time

#Multiprocessing
import multiprocessing
from itertools import repeat

# Useful variables 
solar = 3.826e26 / (4.0*np.pi)

##############################################################################
## Pick the shadow calculation you want to do 
##############################################################################
shadowSingle = False                  # Shadowing for a single asteroid or DEM (multiprocessed)
shadowBinary = True                 # Shadowing for a moving binary (multiprocessed)                 

# shape models
shapeModelPath = "ENTER SHAPE MODEL PATH HERE"  # For single, this is your body. For binary, this is your primary  
secondaryPath = "ENTER SECONDARY SHAPE MODEL PATH HERE" 

# Planet (for solar distance)
planet = None # Enter planet here 
solarDist = planet.rAU # semi major axis in au (assumes circular so need to update)

# Files to save shadows to 
if shadowSingle: 
    shadowFile = "ENTER SHADOW RESULTS FILE HERE" # This is where results will save
    shiftVal = 0.0                                # Not a binary, no shift applied. Do not change 

if shadowBinary: 
    priFile = "ENTER PRIMARY SHADOW RESULTS FILE HERE"
    secFile = "ENTER SECONDARY SHADOW RESULTS FILE HERE"
    
    # Orbit elements
    shiftVal = 2.46     # Sample shift of 1 km (separation between bodies)
    secRot = 0          # Rotation of the secondary (deg.). If a shift is given, this is applied after (moving body around the primary)
    priRot = 0          # Rotation of the primary about its z axis (deg.)


# Enter steps: number of divisions in a 360 degree orbit 
steps = 360


# If including obliquity, add it here
obliq = 0.0 # obliquity in degrees


# Location of the sun 
# NOTES
#   You may need to move this around depending on the orientation or shapes of your DEM 
#   This is preset to standard 3d binaries 
#   At the distances of most asteroids, rays come in approximately parallel, 
#       so additional changes often yield minimal changes to shadows. Semimajor 
#       axis is usually a good starting point 
#   Shadowing products are saved as dot products. This does not affect the strength of 
#       insolation. That is handled later by the orbit module when the model runs 
sLoc = np.asarray([solarDist,0,0]) # Ex: 1AU 


##############################################################################
##############################################################################
## Shadowing Calculation 
## The model should take it from here 
##############################################################################
##############################################################################


##############################################################################
## Read in Shape Models
## DO NOT EDIT
##############################################################################
paths = []
if shadowSingle:
    paths.append(shapeModelPath)

if shadowBinary:
    # Primary
    paths.append(shapeModelPath)
    
    # Secondary
    paths.append(secondaryPath)
    
local = False
shapes = {}
facets = []
allTris = []
shift = (shiftVal,0,0) #The vector in 3d space to shift the secondary. Ex: 2.46 km separation for 1996 FG3
for i in range(len(paths)):
    shapeModelPath = paths[i]
    if i == 1:
        print ("Secondary")
        shapes[i] = shape.shapeModel(filePath = shapeModelPath,local = local,shift = shift,rot = secRot) 
    else: 
        print ("Primary")
        shapes[i] = shape.shapeModel(filePath = shapeModelPath,local = local,rot = priRot)  
    facets.append(len(shapes[i].tris))
    allTris.append(shapes[i].tris)

if len(paths) > 1:
    allTrisJoined = sum(allTris,[])
else:
    allTrisJoined = shapes[0].tris
    
originalShapes = shapes


    
    
##############################################################################
## Single body (not a binary) 
##############################################################################
if shadowSingle:
    print ("MULTIPROCESSED SINGLE BODY SHADOWS")
    open(shadowFile,'w').close()

    shape = shape.shapeModel(shapeModelPath)
    
    # # Time it 
    startTime = time.perf_counter()

    
    # Orientations to cycle through 
    stepSize = 360 / steps
    rotVals = np.arange(steps) * stepSize
    
    
    shadowVals = np.zeros((steps,shape.facetNum))
    orientations = np.linspace(0,360,steps,endpoint = False)
    
    # Orientation of primary 
    iVals = np.arange(np.size(orientations))
    
    # Initialize pool 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print ("pool started")
    
    if obliq == 0.0: 
        # No obliquity
    
        arg1 = orientations
        arg2 = solarDist
        arg3 = shape
    
        result = pool.starmap(shadowMod.CalcShadowsSingleBodyMP,zip(arg1,repeat(arg2),repeat(arg3)))
        res = np.asarray(result)
        
        shadowVals = res
        np.save(shadowFile,shadowVals)
    
    
    else:
        # Obliquity is non zero 
        if obliq != 0.0:
            print ("Obliquity of {} degrees".format(obliq))
            
            arg1 = orientations
            arg2 = solarDist
            arg3 = shape
            arg4 = obliq
            #arg5 = iVals = np.arange(np.size(orientations))
            
            result = pool.starmap(shadowMod.CalcShadowsSingleBodyMP,zip(arg1,repeat(arg2),repeat(arg3), repeat(arg4)))
            res = np.asarray(result)
            shadowVals = res
            np.save(shadowFile,shadowVals)
            
    print ("Shadows Saved to: {}".format(shadowFile))
    
    print ("Time: {}".format(time.perf_counter() - startTime))
    
    
    

##############################################################################
## shadowBinary
##############################################################################    
if shadowBinary:
    
    # Set things up 
    print ("Entered dynamicsMP")
    open(priFile,'w').close()
    open(secFile,'w').close()
    
    # # Time it 
    startTime = time.perf_counter()

    
    # Orientations to cycle through 
    steps = 180
    secStep = 360 / steps
    priStep = 360 / steps
    priRotVals = np.arange(steps) * priStep
    secOrbVals = np.arange(steps) * secStep
    
    
    primaryFluxes = np.zeros((steps,steps,shapes[0].facetNum)) 
    secondaryFluxes = np.zeros((steps,steps, shapes[1].facetNum))
    
    orientations = np.linspace(secStep,360,steps)
    priOrientations = np.linspace(priStep,360,steps)
    
    # Added binary radius for shadow clipping 
    maxBinaryRadius = shapes[0].maxDim + shapes[1].maxDim
    
    # Orientation of primary 
    iVals = np.arange(np.size(priOrientations))
    
    # Initialize pool 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print ("pool started")
    
    arg1 = iVals
    arg2 = priOrientations
    arg3 = steps
    arg4 = secStep
    arg5 = shapes[0]
    arg6 = shapes[1]
    result = pool.starmap(shadowMod.traceShadowsMP,zip(arg1,arg2,repeat(arg3), repeat(arg4),repeat(arg5),repeat(arg6)))
    res = np.asarray(result)
    
    primaryShadows = res[:,0,:,:]
    secondaryShadows = res[:,1,:,:]
    
    # Save to flux files 
    np.save(priFile,primaryShadows)
    np.save(secFile,secondaryShadows)

    print ("Complete")
    print ("Dot Products saved to files")
    
    
    
    
    
    
