#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:14:39 2021

@author: kyso3185

Description: Script for calculating view factors separately from thermal model.
Requires a mesh (ie. obj or wav format). Uses visibility.py to calculate. 

For a single body, this uses three tests. Initially, normals of the facets are 
compared. If more than 90 degrees different, facets cannot see each other and 
visibility is returned as 0. If visibility is not 0, a ray is then traced 
between the two to check for obstructions. If no obstructions, the view factor
between the two triangular facets is calculated


For binary systems, the model iterates through many different orientations of 
the primary and secondary to create a large lookup table that can then be used 
by the model later. Current parameters to scan over are:

Rotation of primary about axis
Rotation of secondary about primary

In the model, these values will be tracked and then lookup table can be used 
to find closest match for each facet at that given timestep.  
"""
import numpy as np
import kdTree3d as kdtree_3d
import visibilityFunctions as vis
import planets
import time 
from scipy import sparse

# Read in Shape Model
import shapeModule as shape

#Multiprocessing
from multiprocessing import Manager
import multiprocessing
import pathos
mp = pathos.helpers.mp
from itertools import repeat
    
    

##############################################################################
##############################################################################
## User Choices 
##############################################################################
##############################################################################
# Option to save as sparse matrix (.npz)
sparseSave = False 

# Choose View Factors Calculation(s) 
singleBodyViewFactors = True 
binaryViewFactors = False

    
if __name__ == '__main__':
        
    if singleBodyViewFactors: 
        ##############################################################################
        ## View factor calculation for solitary body 
        ##############################################################################
        print ("Calculating Single Body View Factors")    
        
        ##############################################################################
        ## Enter the following two file paths 
        ##############################################################################
        shapeModelPath = "SHAPE MODEL FILE PATH HERE"
        vfFile = "VIEW FACTOR STORAGE FILE NAME"
        
        ##############################################################################
        ## Multiprocessed model 
        ## DO NOT ALTER
        ##############################################################################
        local = False
        shape = shape.shapeModel(shapeModelPath,local)
        facets = len(shape.tris)
        
        # KD Tree
        KDTree = kdtree_3d.KDTree(shape.tris,leaf_size = 5, max_depth = 0)
        
        
        # Calculating view factors
        # Establish multiprocessing manager and thread safe storage
        manager = Manager()
        
        iVals = np.arange(0,facets)
        viewFactorStorage = manager.dict()
        infoStorage = manager.dict()
       
        visStartTime = time.perf_counter()
        
        p = mp.Pool(multiprocessing.cpu_count())
        viewFactors = np.zeros((facets,facets))
        viewFactorRun = p.starmap(vis.viewFactorsHelper,zip(repeat(shape.tris),repeat(facets),repeat(KDTree),iVals,repeat(viewFactorStorage),repeat(local)))
        
        
        for i in iVals:
            viewFactors[i,:] = viewFactorStorage[i]
        
        print ("Visibility calculations took: "+str(time.perf_counter()-visStartTime)+" sec")
        print ("Number of final non zero view factors: "+str(np.count_nonzero(viewFactors)))
        
        if sparseSave:
            viewFactors = sparse.csr_matrix(viewFactors)
            sparse.save_npz(vfFile, viewFactors)
            
        
        else: 
            open(vfFile,'w').close()
            np.save(vfFile,viewFactors)
        
        
        
        
    if binaryViewFactors: 
        ###############################################################################
        ## Binary View Factor Calculation 
        ###############################################################################
        print ("Calculating Binary View Factors")  
        
        ###############################################################################
        ## Enter the Following Parameters
        ###############################################################################

        # Shape model paths 
        priModelPath = "PRIMARY SHAPE MODEL FILE (ex: .obj)"
        secModelPath = "SECONDARY SHAPE MODEL FILE (ex: .obj)"
        
        # Files to save results to 
        binaryVFFile = 'BINARY VIEW FACTORS FILE (.npy)' # File to save view factors to (F_ij) 
        flippedVFFile = 'FLIPPED VIEW FACTORS FILE (.npy)' # File to save flipped view factors (F_ji) to
        
        
        # Set parameters for a system 
        planet = planets.FG31996            # Replace with primary body of interest 
        primary = planet 
        secondary = planets.FG31996_Second  # Replace with secondary body of interest
        shiftVal= secondary.separation      # Translational shift to be applied to secondary (km)
        shift = (shiftVal,0,0)              # Moves secondary shape model shiftVal km away in x direction
        steps = 360                         # Number of orientations for secondary (how many divisions in 360 degrees) 
        
        
        
        ###############################################################################
        # Multiprocessed view factor calculation using scan through binary orientations
        # DO NOT ALTER 
        
            # Set the resolution of the scan 
            # Theta is rotation of primary
            # Phi is orbital angle of secondary about primary 
            # Psi is the rotation of secondary about local axis (for non tidally locked 
                # secondaries, not yet implemented)
        ###############################################################################
         
        # Timer start
        startTime = time.perf_counter()
        
        priShape = shape.shapeModel(priModelPath)
        print ("Primary Model Loaded")
        secShape = shape.shapeModel(secModelPath, shift = shift)
        print ("Secondary Model Loaded with a shift of "+str(secondary.separation) + " km")
        
        sysFacetNum = priShape.facetNum + secShape.facetNum
            
        # Define orientations to cycle through 
        # Since secondary is tidally locked, rotate secondary around fixed primary 
        # Will take the difference between secondary phi and primary theta as angle for lookup  
        secStep = 360. / steps
        secOrientations = np.linspace(0,360,steps,endpoint = False)# np.arange(steps) * secStep
        print ("Steps per 360 deg: {}".format(steps))
        print ("First sec orientations: {}".format(secOrientations[:5]))
        
        # Initialize view factor array
        vfLookup = np.zeros((steps,priShape.facetNum,secShape.facetNum))
                            
        # Orientation identifier for multiprocessing  
        iVals = np.arange(np.size(secOrientations))
        
        # Find secondary facets that face the primary 
        dots, whereValid = vis.findVisibleSecondaryFacets(priShape, secShape)
        whereValid = np.ravel(whereValid)
        
        
        # Initialize pool 
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        print ("pool started")
        
        # Do visibility calcs 
        arg1 = priShape
        arg2 = secShape
        arg3 = secOrientations
        arg4 = np.ravel(whereValid)
        arg5 = np.arange(steps)
        result = pool.starmap(vis.vfBinaryMP,zip(repeat(arg1),repeat(arg2),arg3, repeat(arg4), arg5))
        fijRes = np.asarray(result) # View factor results (F_ij)
        print ("Size of result: {}".format(np.shape(fijRes)))
        
        ###############################################################################
        ## Save initial set of view factors (F_ij) to .npy file
        ##    This can be update to .npz if sparse matrices desired 
        ###############################################################################
        np.save(binaryVFFile, fijRes)
        
        
        ###############################################################################
        ## Flip the view factor array (Go from F_ij to F_ji)
        ###############################################################################
        flippedVFArr = np.zeros(np.shape(fijRes))
        total = 0
        for i in range(np.shape(fijRes)[0]):
            flippedVFArr[i],totalTemp = vis.flipVF(fijRes[i],secShape.mesh.areasSqMeters, priShape.mesh.areasSqMeters, priShape.facetNum, secShape.facetNum)
            if totalTemp > total:
                    total = totalTemp
        
        
        np.save(flippedVFFile,flippedVFArr)
        
        # Timer end 
        print ("Total time: "+str(time.perf_counter() - startTime) + " sec")
        print ("Complete")

        
        
        
        
    


