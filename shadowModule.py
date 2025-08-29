#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:14:40 2023

@author: kyso3185

Modification of Shadowing_Test.py for testing shadows on Atlas. Will run 
shadowing for both single (maybe crater) and for a binary.

Might try to generate pngs on atlas but if not it will output arrays of the 
shadows that can be compared against local test cases. 
"""


import numpy as np

# Copy
import copy 

# My Modules
#import Reflection as reflect
import rayTracing as raytracing 
import kdTree3d as kdtree
import shapeModule as shape





##############################################################################
## Functions
##############################################################################

def ShadowCheck(tris,ray,expTri,baseDist):
        # Checks through all triangles to find the closest intersect. Least computationally effective method 
        #distExp = find_distance(tri = expTri,ray = ray)
        hitcount = 0
        closest_tri = expTri
        
        for tri in tris:
            result = tri.intersect_simple(ray)
            if result == 1: #tri.intersect_simple(ray) == 1:
                hitcount += 1
                if tri.num == expTri.num: # If origin triangle
                    continue
                dist = kdtree.find_distance(ray=ray,tri = tri)
                if dist < baseDist:
                    # closer intersect found. Light is stopped before reaching facet
                    # shadowed
                    closest_tri = tri
                    #print ("Shadowed")
                    return True ,tri.num
                elif dist == baseDist:
                    if tri != expTri:
                        print ("diff tri, same distance")
                        continue
                else:
                    # dist is either equal or greater than dist to tri in question. Skip to next
                    # print ("Farther away")
                    #print ("Tri "+str(tri.num)+" is Farther away than expected No. "+str(expTri.num))
                    continue
            if result == 2:
                print ("Something's gone wrong: Ray starting point is in triangle")

        return False,expTri.num    # Light makes it to expected facet, no shadow 
    
def NormVisCheck(tri,ray):
    # first takes the dot product of solar vector with the facet normal
    # Make sure ray is of length 1 
    # Remove
    # Not necessarily remove but be wary of the negative you introduced to deal with standalone landforms (see ray direction line)
    # Make negative if doing single (due to flip of axes) 
    rayDirection = -np.asarray(ray.d / np.linalg.norm(ray.d))
    dotproduct = np.dot(rayDirection,tri.normal)
    i = np.arccos(dotproduct) # solar incidence angle 
    if i <= (np.pi / 2.0) or i >= ((3*np.pi) / 2.0): #if on day side
        # If less than 90 deg, cast ray to see if visible 
        return True, dotproduct # on day side
    return False, dotproduct # on night side


def FluxCalc(dot,dist,solar):
    flux = (solar / (dist**2)) * dot
    return flux

def AdvanceBinaryOrbit(priModel: shape.shapeModel, secModel: shape.shapeModel, priTheta,secPhi):
    # Rotates primary about own axis by a given angle (deg.)
    # Rotates secondary about primary by a given angle (deg.)
    
    # Rotate primary and update vertice/face information 
    priModel.rotateMesh(priTheta)
    
    # Rotates secondary about primary (only for tidally locked secondary)
    secModel.rotateMesh(secPhi)
    
    allTrisNew = []
    allTrisNew.append(priModel.tris)
    allTrisNew.append(secModel.tris)
    
    #allTrisJoinedNew = sum(allTrisNew,[])
    
    return allTrisNew#, allTrisJoinedNew

def clipShadows(priModel: shape, secModel: shape, solarLocation: np.array, maxBinaryRadius):
    # Given the shape models and angle of rotation (primary) and orbit (secondary)
    # determine if it is possible for inter-body shadowing to occur. If it is,
    # Continue with checking shadowing against facets on both bodies. If not, 
    # only check shadowing on same body 
    
    # Find angle between solar direction and placement of secondary 
    vecPriSun = np.asarray(solarLocation - priModel.meshCentroid) # Vector from solar location to primary centroid
    vecPriSec = np.asarray(secModel.meshCentroid - priModel.meshCentroid) # Vector between primary and secondary centroids 
    binarySep = np.linalg.norm(vecPriSec) #Spatial separation between primary and secondary 
    phi = np.arccos((np.dot(vecPriSun, vecPriSec)) / (np.linalg.norm(vecPriSun)* binarySep)) #Angle between 
    # Determine if shadowing is possible 
    if maxBinaryRadius >= binarySep * np.sin(phi):
        # Do shadows if true 
        clip = False
    else: 
        # Only check shadows on own body
        clip = True 
    return clip


def traceShadowsMP(iVal, priOrientation, steps, secSteps, priModel: shape.shapeModel, secModel: shape.shapeModel):#,secondaryFluxes):
    print ("iVal: {}".format(iVal))
    
    # Make deep copies of shapes 
    priModel = copy.deepcopy(priModel)
    secModel = copy.deepcopy(secModel)
    
    
    # Rotate priamry 
    allTris = AdvanceBinaryOrbit(priModel,secModel,priOrientation,0)       
        
    secStep = 360 / secSteps
        
    # primaryFluxSingleIndex = np.zeros((steps,priModel.facetNum))
    # secondaryFluxSingleIndex = np.zeros((steps,secModel.facetNum))
    
    primaryFluxSingleIndex = np.zeros((secSteps,priModel.facetNum))
    secondaryFluxSingleIndex = np.zeros((secSteps,secModel.facetNum))
   
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    #solar = 3.826e26 / (4.0*np.pi)
    
    
    # For a given rotation of the primary, do 360 degree orbit of secondary about the primary 
    # for k in range(steps):
    for k in range(secSteps):
        print ("K: {}".format(k))
        clip = clipShadows(priModel, secModel, sLoc,maxBinaryRadius)
        print ("     Clip: {}".format(clip))
        # Flux Storage
        fluxVals = []
        for i in range(len(allTris)):
            shadowCount = 0
            nonShadowCount = 0
            fluxValsTemp = []
            for tri in allTris[i]:
                sVect = np.asarray(tri.centroid - sLoc)
                baseDist = np.linalg.norm(sVect)
                sVect = sVect * 1/baseDist
                
                # Vis check with sun angle
                ray = raytracing.ray(sLoc,sVect)
                vis = NormVisCheck(tri,ray)              
                if vis[0]:
                    #shadow = ShadowCheck(allTrisJoined,ray,tri,baseDist)
                    #shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                    if not clip: # If clipping function is false and interbody shadowing could happen
                        shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                    if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                        shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                    if shadow[0] == True: 
                        shadowCount += 1
                        fluxValsTemp.append(0)
                    if shadow[0] == False:
                        #flux = FluxCalc(vis[1], baseDist,solar) # remove
                        flux = vis[1]
                        fluxValsTemp.append(flux)
                        nonShadowCount += 1
                else:
                    #visFailCount += 1
                    fluxValsTemp.append(0) # remove
            #if i == 0:
            #    print ("     Primary shadow count: "+str(shadowCount))
            #else: 
            #    print("     Secondary Shadow Count: "+str(shadowCount))
            fluxVals.append(fluxValsTemp)

        
        # Store current orientation 
        # Store mutiples
        #primaryFluxes[j][k] = np.asarray(fluxVals[0])
        #secondaryFluxes[j][k] = np.asarray(fluxVals[1])
            
        # Store fluxes
        primaryFluxSingleIndex[k] = np.asarray(fluxVals[0])
        secondaryFluxSingleIndex[k] = np.asarray(fluxVals[1]) 
    
    
        # # Advance binary orbit 
        # #allTris, allTrisJoined = AdvanceBinaryOrbit(shapes[0],shapes[1],priStep,secStep)
            
        
        #print ("Primary rotated by "+str(priStep) + " degrees")
        #print ("Secondary orbited around "+str(secStep) + " degrees")
        #print ("     Brute force shadows took: "+str(time.perf_counter()-bruteST)+" sec")
        
        
        # Only rotate the secondary
        allTris = AdvanceBinaryOrbit(priModel,secModel,0,secStep)
        print ("Secondary to {} degrees".format(secStep * k))#orientations[k]))
    
    return primaryFluxSingleIndex,secondaryFluxSingleIndex


def traceShadowsMP(iVal, priOrientation, steps, secStep, priModel: shape, secModel: shape, maxBinaryRadius = None):#,secondaryFluxes):
    print ("iVal: {}".format(iVal))
    # Rotate priamry 
    allTris = AdvanceBinaryOrbit(priModel,secModel,priOrientation,0)
    primaryFluxSingleIndex = np.zeros((steps,priModel.facetNum))
    secondaryFluxSingleIndex = np.zeros((steps,secModel.facetNum))
   
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    
    # Determine maximum binary radius if not already calculated 
    # Added binary radius for shadow clipping 
    if maxBinaryRadius == None:
        maxBinaryRadius = priModel.maxDim + secModel.maxDim
    
    # For a given rotation of the primary, do 360 degree orbit of secondary about the primary 
    for k in range(steps):
            clip = clipShadows(priModel, secModel, sLoc,maxBinaryRadius)
            print ("     Clip: {}".format(clip))
            # Flux Storage
            fluxVals = []
            for i in range(len(allTris)):
                shadowCount = 0
                nonShadowCount = 0
                fluxValsTemp = []
                for tri in allTris[i]:
                    sVect = np.asarray(tri.centroid - sLoc)
                    baseDist = np.linalg.norm(sVect)
                    sVect = sVect * 1/baseDist
                    
                    # Vis check with sun angle
                    ray = raytracing.ray(sLoc,sVect)
                    vis = NormVisCheck(tri,ray)              
                    if vis[0]:
                        if not clip: # If clipping function is false and interbody shadowing could happen
                            shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                        if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                            shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                        if shadow[0] == True: 
                            shadowCount += 1
                            fluxValsTemp.append(0)
                        if shadow[0] == False:
                            flux = vis[1]
                            fluxValsTemp.append(flux)
                            nonShadowCount += 1
                    else:
                        fluxValsTemp.append(0) 
                if i == 0:
                    print ("     Primary shadow count: "+str(shadowCount))
                else: 
                    print("     Secondary Shadow Count: "+str(shadowCount))
                fluxVals.append(fluxValsTemp)
                
            # Store fluxes
            primaryFluxSingleIndex[k] = np.asarray(fluxVals[0])
            secondaryFluxSingleIndex[k] = np.asarray(fluxVals[1]) #Manager array
            
            # Only rotate the secondary
            allTris = AdvanceBinaryOrbit(priModel,secModel,0,secStep)
            print ("Secondary to {} degrees".format(orientations[k]))
    
    return primaryFluxSingleIndex,secondaryFluxSingleIndex



def traceShadowsHighResEclipseMP(priOrientation, secOrientations, secOrientationDiff, priModel: shape.shapeModel, secModel: shape.shapeModel):#,secondaryFluxes):
    #print ("iVal: {}".format(iVal))
    
    # Make deep copies of shapes 
    priModel = copy.deepcopy(priModel)
    secModel = copy.deepcopy(secModel)
    
    
    # Rotate priamry 
    allTris = AdvanceBinaryOrbit(priModel,secModel,priOrientation,0)
        
        
    # secStep = 360 / secSteps
        
    
    primaryFluxSingleIndex = np.zeros((np.shape(secOrientations)[0],priModel.facetNum))
    secondaryFluxSingleIndex = np.zeros((np.shape(secOrientations)[0],secModel.facetNum))
   
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    #solar = 3.826e26 / (4.0*np.pi)
    
    
    # For a given rotation of the primary, iterate through preset secondary orbital orientations 
    for k in range(np.shape(secOrientations)[0]):
    #for k in range(secSteps):
        print ("K: {}".format(k))
        #clip = clipShadows(priModel, secModel, sLoc,maxBinaryRadius)
        #print ("     Clip: {}".format(clip))
        # Flux Storage
        fluxVals = []
        for i in range(len(allTris)):
            shadowCount = 0
            nonShadowCount = 0
            fluxValsTemp = []
            for tri in allTris[i]:
                sVect = np.asarray(tri.centroid - sLoc)
                baseDist = np.linalg.norm(sVect)
                sVect = sVect * 1/baseDist
                
                # Vis check with sun angle
                ray = raytracing.ray(sLoc,sVect)
                vis = NormVisCheck(tri,ray)              
                if vis[0]:
                    #shadow = ShadowCheck(allTrisJoined,ray,tri,baseDist)
                    #shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                    # if not clip: # If clipping function is false and interbody shadowing could happen
                    shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                    # if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                    #     shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                    if shadow[0] == True: 
                        shadowCount += 1
                        fluxValsTemp.append(0)
                    if shadow[0] == False:
                        #flux = FluxCalc(vis[1], baseDist,solar) # remove
                        flux = vis[1]
                        fluxValsTemp.append(flux)
                        nonShadowCount += 1
                else:
                    #visFailCount += 1
                    fluxValsTemp.append(0) # remove

            fluxVals.append(fluxValsTemp)

            
        # Store fluxes
        primaryFluxSingleIndex[k] = np.asarray(fluxVals[0])
        secondaryFluxSingleIndex[k] = np.asarray(fluxVals[1]) 
        
        
        # # Only rotate the secondary
        # allTris = AdvanceBinaryOrbit(priModel,secModel,0,secStep)
        # print ("Secondary to {} degrees".format(secStep * k))#orientations[k]))
        
        # Only rotate the secondary
        if k < np.shape(secOrientationDiff)[0]: # diff is one element shorter than orientations
            allTris = AdvanceBinaryOrbit(priModel,secModel,0,secOrientationDiff[k])
            print ("Secondary to {} degrees".format(secOrientationDiff[k] * k))#orientations[k]))
    
    return primaryFluxSingleIndex,secondaryFluxSingleIndex



def binarySingleOrientationShadows(priModel: shape.shapeModel, secModel: shape.shapeModel, priTheta, secPhi):
    
    
    # if priTheta != 0 or secPhi != 0:
    #     # Make deep copies of shapes 
    #     priModel = copy.deepcopy(priModel)
    #     secModel = copy.deepcopy(secModel)
        
        
    # Rotate shapes with desired theta and phi 
    allTris = AdvanceBinaryOrbit(priModel,secModel,priTheta,secPhi)
        
    # # Flux storage arrays 
    # primaryFluxes = np.zeros((secSteps,priModel.facetNum))
    # secondaryFluxes = np.zeros((secSteps,secModel.facetNum))
   
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    #solar = 3.826e26 / (4.0*np.pi)
    
    
    # Calculate shadows for given orientation 
    # print ("K: {}".format(k))
    # clip = clipShadows(priModel, secModel, sLoc,maxBinaryRadius)
    # print ("     Clip: {}".format(clip))
    # Flux Storage
    fluxVals = []
    for i in range(len(allTris)): # For each body (i is 0 or 1)
        shadowCount = 0
        nonShadowCount = 0
        fluxValsTemp = []
        for tri in allTris[i]: # Iterate through everything on that body 
            sVect = np.asarray(tri.centroid - sLoc)
            baseDist = np.linalg.norm(sVect)
            sVect = sVect * 1/baseDist
            
            # Vis check with sun angle
            ray = raytracing.ray(sLoc,sVect)
            vis = NormVisCheck(tri,ray)              
            if vis[0]:
                # if not clip: # If clipping function is false and interbody shadowing could happen
                shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                # if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                #     shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                if shadow[0] == True: 
                    shadowCount += 1
                    fluxValsTemp.append(0)
                if shadow[0] == False:
                    #flux = FluxCalc(vis[1], baseDist,solar) # remove
                    flux = vis[1]
                    fluxValsTemp.append(flux)
                    nonShadowCount += 1
            else:
                #visFailCount += 1
                fluxValsTemp.append(0) # remove
        #if i == 0:
        #    print ("     Primary shadow count: "+str(shadowCount))
        #else: 
        #    print("     Secondary Shadow Count: "+str(shadowCount))
        fluxVals.append(fluxValsTemp)

        
    # Store current orientation 
    # Store mutiples
    #primaryFluxes[j][k] = np.asarray(fluxVals[0])
    #secondaryFluxes[j][k] = np.asarray(fluxVals[1])
         
                
    # Store fluxes
    primaryFluxes = np.asarray(fluxVals[0])
    secondaryFluxes = np.asarray(fluxVals[1]) 

    
    return primaryFluxes,secondaryFluxes

def CalcShadowsSingleBody(steps, stepSize,shape):
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    #solar = 3.826e26 / (4.0*np.pi)
    
    
    # For a given rotation of the primary, do 360 degree orbit of secondary about the primary 
    # for k in range(steps):
    shadowVals = np.zeros((steps, shape.facetNum))
    for k in range(steps):
        print ("K: {}".format(k))
        # Flux Storage
        #fluxVals = []
        #for i in range(len(shape.tris)):
        shadowCount = 0
        nonShadowCount = 0
        #fluxValsTemp = []
        
        # sVects = shape.centroids - sLoc
        # baseDists = np.linalg.norm(sVects)
        # sVects = sVects * (1 / baseDists)
        
        # vis = NormVisCheck(tri, ray)
        # rayDirection = -np.asarray(ray.d / np.linalg.norm(ray.d))
        # dotproduct = np.dot(rayDirection,tri.normal)
        # i = np.arccos(dotproduct) # solar incidence angle 
        # if i <= (np.pi / 2.0) or i >= ((3*np.pi) / 2.0): #if on day side
        #     # If less than 90 deg, cast ray to see if visible 
        #     return True, dotproduct # on day side
        # return False, dotproduct # on night side
    
        # directions = - sVects
        # #dotproducts = np.dot(directions,shape.trimesh_normals)
        # dotproducts = np.einsum('ij,ij->i',directions,shape.trimesh_normals)
        # incidenceAngles = np.arccos(dotproducts)


        # print (np.count_nonzero(incidenceAngles))

        for tri in shape.tris:
            i = tri.num
            sVect = np.asarray(tri.centroid - sLoc)
            baseDist = np.linalg.norm(sVect)
            sVect = sVect * 1/baseDist
            
            # Vis check with sun angle
            ray = raytracing.ray(sLoc,sVect)
            vis = NormVisCheck(tri,ray)              
            if vis[0]:
            #if incidenceAngles[i] != 0:
                sVect = np.asarray(tri.centroid - sLoc)
                baseDist = np.linalg.norm(sVect)
                sVect = sVect * 1/baseDist
                #ray = raytracing.ray(sLoc,sVects[i])
                ray = raytracing.ray(sLoc,sVect)
                shadow = ShadowCheck(shape.tris,ray,tri,baseDist)
                if shadow[0] == True: 
                    shadowCount += 1
                    #fluxValsTemp.append(0)
                    shadowVals[k][i] = 0.
                if shadow[0] == False:
                    #flux = FluxCalc(vis[1], baseDist,solar) # remove
                    #flux = vis[1]
                    #fluxValsTemp.append(flux)
                    # shadowVals[k][i] = dotproducts[i]#flux
                    shadowVals[k][i] = vis[1]
                    nonShadowCount += 1
                # else:
                #     #visFailCount += 1
                #     fluxValsTemp.append(0) # remove
                # No else statement required. Already a zero array 
                    
            # #if i == 0:
            # #    print ("     Primary shadow count: "+str(shadowCount))
            # #else: 
            # #    print("     Secondary Shadow Count: "+str(shadowCount))
            # fluxVals.append(fluxValsTemp)
    
        shape.rotateMesh(stepSize)
        print("After Shadowing: {}".format(np.count_nonzero(shadowVals[k])))
        print ("Secondary to {} degrees".format(stepSize * (k+1)))
        print (shape.tris[0].normal)
        
        # # Store current orientation 
        # # Store mutiples
        # #primaryFluxes[j][k] = np.asarray(fluxVals[0])
        # #secondaryFluxes[j][k] = np.asarray(fluxVals[1])
            
        # # Store fluxes
        # primaryFluxSingleIndex[k] = np.asarray(fluxVals[0])
        # secondaryFluxSingleIndex[k] = np.asarray(fluxVals[1]) 
    
    
        # # Advance binary orbit 
        # #allTris, allTrisJoined = AdvanceBinaryOrbit(shapes[0],shapes[1],priStep,secStep)
            
        
        #print ("Primary rotated by "+str(priStep) + " degrees")
        #print ("Secondary orbited around "+str(secStep) + " degrees")
        #print ("     Brute force shadows took: "+str(time.perf_counter()-bruteST)+" sec")
        
        
        # # Only rotate the secondary
        # allTris = AdvanceBinaryOrbit(priModel,secModel,0,secStep)
        # print ("Secondary to {} degrees".format(secStep * k))
    return shadowVals


def CalcShadowsSingleBodyWithFixedObliquity(obliq, steps, stepSize,shape, solarDist):
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    #sLoc = np.asarray([2.124e11,0,0])
    sLoc = np.asarray([solarDist*1.496e11,0,0])
    #solar = 3.826e26 / (4.0*np.pi)
    
    
    # For a given rotation of the primary, do 360 degree orbit of secondary about the primary 
    # for k in range(steps):
    shadowVals = np.zeros((steps, shape.facetNum))
    for k in range(steps):
        print ("K: {}".format(k))
        # Flux Storage
        #fluxVals = []
        #for i in range(len(shape.tris)):
        shadowCount = 0
        nonShadowCount = 0
        #fluxValsTemp = []
        
        # sVects = shape.centroids - sLoc
        # baseDists = np.linalg.norm(sVects)
        # sVects = sVects * (1 / baseDists)
        
        # vis = NormVisCheck(tri, ray)
        # rayDirection = -np.asarray(ray.d / np.linalg.norm(ray.d))
        # dotproduct = np.dot(rayDirection,tri.normal)
        # i = np.arccos(dotproduct) # solar incidence angle 
        # if i <= (np.pi / 2.0) or i >= ((3*np.pi) / 2.0): #if on day side
        #     # If less than 90 deg, cast ray to see if visible 
        #     return True, dotproduct # on day side
        # return False, dotproduct # on night side
    
        # directions = - sVects
        # #dotproducts = np.dot(directions,shape.trimesh_normals)
        # dotproducts = np.einsum('ij,ij->i',directions,shape.trimesh_normals)
        # incidenceAngles = np.arccos(dotproducts)


        # print (np.count_nonzero(incidenceAngles))

        for tri in shape.tris:
            i = tri.num
            sVect = np.asarray(tri.centroid - sLoc)
            baseDist = np.linalg.norm(sVect)
            sVect = sVect * 1/baseDist
            
            # Vis check with sun angle
            ray = raytracing.ray(sLoc,sVect)
            vis = NormVisCheck(tri,ray)              
            if vis[0]:
            #if incidenceAngles[i] != 0:
                sVect = np.asarray(tri.centroid - sLoc)
                baseDist = np.linalg.norm(sVect)
                sVect = sVect * 1/baseDist
                #ray = raytracing.ray(sLoc,sVects[i])
                ray = raytracing.ray(sLoc,sVect)
                shadow = ShadowCheck(shape.tris,ray,tri,baseDist)
                if shadow[0] == True: 
                    shadowCount += 1
                    #fluxValsTemp.append(0)
                    shadowVals[k][i] = 0.
                if shadow[0] == False:
                    #flux = FluxCalc(vis[1], baseDist,solar) # remove
                    #flux = vis[1]
                    #fluxValsTemp.append(flux)
                    # shadowVals[k][i] = dotproducts[i]#flux
                    shadowVals[k][i] = vis[1]
                    nonShadowCount += 1

        
        shape.rotateWithObliquity(obliq=obliq, rot=stepSize)
        print("After Shadowing: {}".format(np.count_nonzero(shadowVals[k])))
        print ("Secondary to {} degrees".format(stepSize * (k+1)))
        print (shape.tris[0].normal)
        
        # # Store current orientation 
        # # Store mutiples
        # #primaryFluxes[j][k] = np.asarray(fluxVals[0])
        # #secondaryFluxes[j][k] = np.asarray(fluxVals[1])
            
        # # Store fluxes
        # primaryFluxSingleIndex[k] = np.asarray(fluxVals[0])
        # secondaryFluxSingleIndex[k] = np.asarray(fluxVals[1]) 
    
    
        # # Advance binary orbit 
        # #allTris, allTrisJoined = AdvanceBinaryOrbit(shapes[0],shapes[1],priStep,secStep)
            
        
        #print ("Primary rotated by "+str(priStep) + " degrees")
        #print ("Secondary orbited around "+str(secStep) + " degrees")
        #print ("     Brute force shadows took: "+str(time.perf_counter()-bruteST)+" sec")
        
        
        # # Only rotate the secondary
        # allTris = AdvanceBinaryOrbit(priModel,secModel,0,secStep)
        # print ("Secondary to {} degrees".format(secStep * k))
    return shadowVals




def CalcShadowsSingleBodyWithObliquity(obliq, steps, stepSize,startShape, solarLocation = np.asarray([2.124e11,0,0])):
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    
    # This will change with each call. Probably just feed in the different 
    # Solar location vectors and then do the normal rotation 
    # sLoc = np.asarray([2.124e11,0,0])
    #solar = 3.826e26 / (4.0*np.pi)
    sLoc = solarLocation
    
    # Make deep copies of shapes 
    shape = copy.deepcopy(startShape)
    
    # For a given rotation of the primary, do 360 degree orbit of secondary about the primary 
    # for k in range(steps):
    shadowVals = np.zeros((steps, shape.facetNum))
    for k in range(steps):
        #print ("K: {}".format(k))
        # Flux Storage
        #fluxVals = []
        #for i in range(len(shape.tris)):
        shadowCount = 0
        nonShadowCount = 0

        for tri in shape.tris:
            i = tri.num
            sVect = np.asarray(tri.centroid - sLoc)
            baseDist = np.linalg.norm(sVect)
            sVect = sVect * 1/baseDist
            
            # Vis check with sun angle
            ray = raytracing.ray(sLoc,sVect)
            vis = NormVisCheck(tri,ray)              
            if vis[0]:
            #if incidenceAngles[i] != 0:
                sVect = np.asarray(tri.centroid - sLoc)
                baseDist = np.linalg.norm(sVect)
                sVect = sVect * 1/baseDist
                #ray = raytracing.ray(sLoc,sVects[i])
                ray = raytracing.ray(sLoc,sVect)
                shadow = ShadowCheck(shape.tris,ray,tri,baseDist)
                if shadow[0] == True: 
                    shadowCount += 1
                    #fluxValsTemp.append(0)
                    shadowVals[k][i] = 0.
                if shadow[0] == False:
                    #flux = FluxCalc(vis[1], baseDist,solar) # remove
                    #flux = vis[1]
                    #fluxValsTemp.append(flux)
                    # shadowVals[k][i] = dotproducts[i]#flux
                    shadowVals[k][i] = vis[1]
                    nonShadowCount += 1

        
        shape.rotateWithObliquity(obliq=obliq, rot=stepSize)
        print("K:  " +str(k)+", After Shadowing: {}".format(np.count_nonzero(shadowVals[k])))
        #print ("Rotation to {} degrees".format(stepSize * (k+1)))
        print (shape.tris[0].normal)
        

    return shadowVals

def CalcShadowsSingleBodyMP(orientation,solarDist, startShape, obliq = None):
    # Multiprocessed version of CalcShadowsSingleBody. Each different rotation
    #    sent to a different core 
    
    # Make deep copies of shape
    shape = copy.deepcopy(startShape)
    
    # Rotate shape according to the orientation we want 
    if obliq == None: # If no obliquity
        shape.rotateMesh(orientation)
    else: # If obliquity 
        shape.rotateWithObliquity(obliq, rot = orientation) 
    
    
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([solarDist*1.496e11,0,0])
    #solar = 3.826e26 / (4.0*np.pi)
    
    
    # For a given rotation of the primary, do 360 degree orbit of secondary about the primary 
    # for k in range(steps):
    shadowVals = np.zeros(shape.facetNum)
    print ("Orientation: {}".format(orientation))
    
    shadowCount = 0
    nonShadowCount = 0

    for tri in shape.tris:
        i = tri.num
        sVect = np.asarray(tri.centroid - sLoc)
        baseDist = np.linalg.norm(sVect)
        sVect = sVect * 1/baseDist
        
        # Vis check with sun angle
        ray = raytracing.ray(sLoc,sVect)
        vis = NormVisCheck(tri,ray)              
        if vis[0]:
        #if incidenceAngles[i] != 0:
            sVect = np.asarray(tri.centroid - sLoc)
            baseDist = np.linalg.norm(sVect)
            sVect = sVect * 1/baseDist
            #ray = raytracing.ray(sLoc,sVects[i])
            ray = raytracing.ray(sLoc,sVect)
            shadow = ShadowCheck(shape.tris,ray,tri,baseDist)
            if shadow[0] == True: 
                shadowCount += 1
                #fluxValsTemp.append(0)
                shadowVals[i] = 0.
            if shadow[0] == False:
                #flux = FluxCalc(vis[1], baseDist,solar) # remove
                #flux = vis[1]
                #fluxValsTemp.append(flux)
                # shadowVals[k][i] = dotproducts[i]#flux
                shadowVals[i] = vis[1]
                nonShadowCount += 1


    print("After Shadowing: {}".format(np.count_nonzero(shadowVals)))
        

    return shadowVals


