#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:29:58 2022

@author: kyso3185

Run script used to call and run the BTM for either a standalone single body
(or DEM) or a binary asteroid. 
"""

# Imports
import numpy as np
import time

# Planetary libraries
import planets
import heat3d_Single as heat3d # If you were to do run a single body model, use this script (similar process, but simpler implementation)
import heat3d_Binary as bheat3d



if __name__ == '__main__':
    startTime = time.perf_counter()
    
    #################################
    # Choose Model Mode
    #################################
    single = False  # Single asteroid or DEM 
    binary = True   # Interacting binary asteroid 
    
    
    ###########################################################################
    ###########################################################################
    # Single Body or DEM Mode
    ###########################################################################
    ###########################################################################
    if single:
        print ("Mode Selected: Single Body Model")
        print ("-----------------------------------------------------------")
        
        #################################
        # Initialization
        #################################
       
        # Choose a planet or small body
        #   Check if the body is in the planets.py repository
        p = planets.Europa
        
        # Give links to pertinent files
        #   shapeModelPath is the file path to the shape or DEM being used
        #   vfFile is the view factor file. If no view factors, enter 'None'
        #   shadowPath is the shadowing file for the shaoe or DEM of interest
        #   outFile is the file to store output results    
        shapeModelPath = 'INSERT SHAPE MODEL OR DEM PATH HERE'
        vfFile = 'INSERT VIEW FACTOR FILE HERE' 
        shadowPath = 'INSERT SHADOWING FILE HERE'
        outFile = 'INSERT FILE NAME FOR OUTPUT RESULTS'
        
        # Create Model
        #   Required variables: 
        #       shapeModelPath: shapeModelPath above
        #       planet: planets.py entry p above 
        #       shadowLookupPath: shadowPath above
        #       ndays: number of days to output after equilibration
        m = heat3d.SingleModel(shapeModelPath = shapeModelPath,planet = p,local = False,shadowLookupPath = shadowPath,vfPath = vfFile)
        print ("Model initialized")
    
        #################################
        # Model Execution 
        #################################
        
        # Run Model
        #   If including view factors, enter frequency of view factor calculation (every n steps) 
        m.run(vfFreq=1)
        
        # Save results 
        print ("Saving all temps")
        np.save(outFile, m.T)
        print ("saved to {}".format(outFile))
        print ("Complete")
        
        
        
        
        
        
        
        
    ###########################################################################
    ###########################################################################
    # Binary Model Mode
    ###########################################################################
    ###########################################################################
    if binary:
        print ("Mode Selected: Binary Thermal Model")
        print ("-----------------------------------------------------------")
        
        #################################
        # Initialization
        #################################
        primary = None              # Pick a body for the primary  
        secondary = None            # Pick a body for the secondary 
        
        
        #################################
        # Other Variables
        #################################
        calcBYORP = False           # Calculate forces and accelerations during readout period 
        calcShadows = True          # Perform raytracing during model to give exact shadows during readout 
        noBinaryVF = True           # Do not include scattering between primary and secondary aka moonshine
        rayTraceDivisions = None    # Max number of times you'll do on the fly ray tracing per orbit (if calcShadows = True)
        noEclipses = True           # Whether to include binary interaction in the form of eclipses 
        vfFreq = 1                  # How often to update the radiation from view factors (time bottleneck)
        vfOff = True                # Whether to do any form of view factors, including single body or binary 
        secondaryShift = (1.0,0,0)  # Shift of the secondary away from primary
        
        
        # Binary Interaction 
        #######################################################################
        # Shadows 180 x 180
        priShadows = 'PRIMARY SHAODWS FILE HERE (.npy)'
        secShadows = 'SECONDARY SHAODWS FILE HERE (.npy)'
        
        # Shape models 
        priShapePath = 'PRIMARY SHAPE MODEL PATH HERE'
        secShapePath = 'SECONDARY SHAPE MODEL PATH HERE'
        
        # Rotation added to secondary shape model (if none, say 0)
        shapeModelRotation = 0 # [degrees]
        
        # View Factors 
        #   priVF, secVF are the single body view factors for the primary and secondary 
        #   binaryVF: Moonshine view factor lookup table (effectively F_ij)
        #   binaryFlippedVF: reciprocal Moonshine view factors (effectively F_ji)
        #   If view factors not included put 'None'
        #   If want single body view factors but no moonshine, only set binaryVF, 
        #       binaryFlippedVF to 'None'
        priVF = 'PRIMARY VIEW FACTORS HERE (.npy)'
        secVF = 'SECONDARY VIEW FACTORS HERE (.npy)'
        binaryVF = None
        binaryFlippedVF = None
        
        # Save Files 
        #######################################################################
        outPositionFile =  'POSITIONS OF BINARY DURING OUTPUT FILE PATH HERE (.npy)'
        outPriTemps = 'PRIMARY TEMPERATURES OUTPUT FILE HERE (.npy)'
        outSecTemps = 'SECONDARY TEMPERATURES OUTPUT FILE HERE (.npy)'
        byorpFile =   'BYORP OUTPUT HERE (IF INCLUDED) (.npy)'
        
        

        
        #################################
        ## Run
        #################################
        startTime = time.perf_counter()
        
        # Create Model 
        m = bheat3d.binaryModel(priShapeModelPath = priShapePath, secShapeModelPath = secShapePath, priPlanet = primary, \
                                secPlanet = secondary,priShadowLookupPath = priShadows, secShadowLookupPath = secShadows,\
                                    priVF = priVF, secVF = secVF,binaryVF = binaryVF,flippedBinaryVF = binaryFlippedVF, noBVF = noBinaryVF, \
                                        AdaptiveShadows = calcShadows, shapeModelRotation = shapeModelRotation, calcBYORP=calcBYORP, noEclipses=noEclipses, \
                                           secondaryShift = secondaryShift)
        # Run Model 
        m.run(vfFreq = vfFreq, vfOff = vfOff, rayTraceDivisions=rayTraceDivisions)

        
        # Entire Temperature Array 
        np.save(outPriTemps,m.priT)
        np.save(outSecTemps,m.secT)
        
        print ("saved to {}".format(outPriTemps))
        print ("saved to {}".format(outSecTemps))
        
        # Positions (Time, theta, phi, r, flux)
        # These are used to match results to the gemoetry that the system was in
        #   at each timestep 
        np.save(outPositionFile,m.lt)
        
        print ("-----------------------------------------------------------")
        
        #################################
        ## BYORP 
        ## Not part of Thermal Model
        #################################
        if calcBYORP:
            netForceVectors = m.netForceVectorStorage
            
            np.save(byorpFile,netForceVectors)
            
            print ("Net force vectors saved to {}".format(byorpFile))
        
        print ("-----------------------------------------------------------")  
        print ("Complete")
        

    
        
    totalTime = time.perf_counter() - startTime 
    print ("Total time: {}".format(totalTime))
    



