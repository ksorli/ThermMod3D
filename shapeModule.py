#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 12:48:24 2021

@author: kyso3185

This script contains the setup for the "shapeModel" class. It contains 
functions to read in a .obj file and store the vertex and face information. 
It will calculate normals and find the dot product of each of those normals 
rotated around a circle given a solar direction vector.
It will also save information used to plot and analyze the shape models. 

this is a later iteration of the Shape model class, which is capable of handling
multiple bodies and shifting their coordinates.

The loadOBJ_Trimesh function utilizes the trimesh module, which can be installed 
with Conda. 
"""

# Imports 
import numpy as np
import trimesh
import math
from rayTracing import triangle

class shapeModel():
    
    # Initialization
    def __init__(self, filePath, sVect=[1,0,0],local = False, shift = (0,0,0),rot = 0,obliq = 0, shapeModelRotation = 0):
        # filePath: path to shape file 
        # sVect: direction sun is coming from (can be changed) 
        # local: Can be invoked to change direction of the sun for some local DEM (non-whole body) shapes 
        # rot: rotation (degrees) to be applied after initialization (ex: move secondary around primary)
        # obliq: obliquity (degrees) to be applied while moving the shape 
        # shapeModelRotation: base rotation in degrees to add to a shape model 
        
        
        self.verts = []             # List of vertices 
        self.faces = {}             # Dictionary of faces
        self.path = filePath        # Path to the shape file 
        self.normals = {}           # Dictionary of face normals (updates) 
        self.startingNormals = {}   # Dictionary of face normals before rotation added 
        self.rotNorm = {}           # Dictionary of rotated face normals 
        self.tris = []              # Raytracing triangle objects 
        
        self.loadOBJ_Trimesh(self.path,local,shapeModelRotation,shift,rot,obliq) # Imports shape 
        self.findNormals()
        self.lats = np.asarray([tri.lat for tri in self.tris]) # Latitudes for each triangle
        self.longs = np.asarray([tri.long for tri in self.tris]) # Longitudes for each triangle
        
        

    
    def loadOBJ_Trimesh(self,filePath,local = False,shapeModelRotation = 0, shift = (0,0,0),rot = 0,obliq = 0): 
        # Imports shape (usually .obj) using trimesh 
        self.mesh = trimesh.load_mesh(filePath)
        tri_mesh = self.mesh
        
        if shapeModelRotation != 0:
            # Rotate the base shape model. Sometimes required for secondaries in binary systems 
            print ("Rotating base shape model by {} degrees".format(shapeModelRotation))
            rotate = makeRotationMatrix(shapeModelRotation, 'z')
            tri_mesh.apply_transform(rotate)
        else: print ("No rotation added to base shape model")
        
        if shift != (0,0,0):
            # Applies translational shift 
            print ("Applying shift of "+str(shift))
            tri_mesh.apply_translation(shift)
            
        if rot != 0:
            # If shifted, this will rotate with shift, like a secondary about a primary 
            print ("Applying rotation of "+str(rot) +" degrees")
            # Note: this is rotation about the z axis 
            rotate = makeRotationMatrix(rot, 'z')
            tri_mesh.apply_transform(rotate)
           
        if obliq != 0:
            # Note that obliquity shift introduces a rotation about the y axis 
            if local == True:
                print ("Are you sure you want to introduce obliquity to a local object? You can, but make sure to double check your geometry")
            print ("Applying obliquity of {} degrees".format(obliq))
            self.obliquityShift(obliq,start = True)
            

        # Spatial properties of mesh and other useful things 
        self.meshCentroid = self.mesh.centroid      # Centroid of bulk mesh 
        self.maxDim = np.max(self.mesh.extents) / 2.    # Largest extent of the shape model 
        self.vertices = np.asarray(tri_mesh.vertices)   # Array of vertices 
        self.facets = tri_mesh.faces                # Gives the vertice numbers associated with a facet 
        self.facetNum = self.facets.shape[0]        # Number of facets 
        self.triangles = tri_mesh.triangles         # triangle objects from trimesh 
        self.areas = tri_mesh.area_faces            # Areas of the faces in square km 
        self.areasSqMeters = self.areas * 1e6       # Areas of the faces in square meters
        self.centroids = tri_mesh.triangles_center  # centroids of the facets 

        
        if local: # If this is a local feature (i.e. a crater) instead of a global body, then shift the shape model
            # Note: local has largely become nonessential with addition of rotation of shape model 
            #   However, it is included in case users find this more intuitive 
            self.vertices[:,[2,0]] = self.vertices[:,[0,2]] # Swap x and z vertices 
        
        # Array form of normals
        self.trimesh_normals = np.asarray(tri_mesh.face_normals)
        
        facet_dict = {}
        for i in range(self.facetNum):
            self.normals[i] = tri_mesh.face_normals[i]
            facet_dict[i] = np.asarray((self.vertices[self.facets[i][0]],self.vertices[self.facets[i][1]],self.vertices[self.facets[i][2]]))
        self.faces = facet_dict
    
    def shiftMesh(self,shift):
        # Shifting mesh after initialization 
        print ("Separate shift of {} being applied".format(shift))
        self.mesh.apply_translation(shift)
        self.update()
    
    def rotateMesh(self,angle):
        # Rotates mesh a given angle about the z axis 
        # angle is in degrees
        theta = np.radians(angle)
        rotate = np.asarray([np.cos(theta),-np.sin(theta),0,0,np.sin(theta),np.cos(theta),0,0,0,0,1,0,0,0,0,1]).reshape((4,4))
        self.mesh.apply_transform(rotate)
        self.update()
        
    def update(self):
        # Called by the model and ray tracing scripts to update the position of the mesh  
        mesh = self.mesh
        self.tris = []
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = mesh.triangles
        self.trimesh_normals = mesh.face_normals
        self.centroids = mesh.triangles_center
        self.meshCentroid = mesh.centroid
        for i in range(self.facetNum):
            self.normals[i] = mesh.face_normals[i]
            self.faces[i] = np.asarray((self.vertices[self.facets[i][0]],self.vertices[self.facets[i][1]],self.vertices[self.facets[i][2]]))
            v1,v2,v3 = self.faces.get(i)
            self.tris.append(triangle(v1,v2,v3,self.normals[i],i))
        
            
    def findNormals(self):
        # Custom way of storing normals
        for f in self.faces:
            v1,v2,v3 = self.faces.get(f)
            vector1 = [v1-v2]
            vector2 = [v1-v3]
            n = np.cross(vector1,vector2) #unnormalized normal
            normal = normalize(n[0])
            self.startingNormals[f] = [normal]
            self.tris.append(triangle(v1,v2,v3,normal, f))
    
                
    def rotateNormalsByAngle(self,angle = 0):
        # Rotates normals by a predetermined angle
        # added Sept 10, 2021 during debugging but will probably stay, as it is useful
        theta = np.radians(angle)
        for n in self.startingNormals.keys():
            #rotate the normals and store the rotated version 
            newN = np.asarray(rotate(self.startingNormals[n][0],theta))
            self.normals[n] = newN
            
    
    def dotProductWithVector(self,sVect = np.asarray([1,0,0])):
        # Find dot product of normal vectors with given vectors
        # Clip to 0 if more than 90 deg (like sun is below horizon)
        dotProducts = np.dot(self.trimesh_normals, sVect)
        dotProducts[dotProducts <= 0] =  0
        return dotProducts
    
    def obliquityShift(self,obliq,start = False):
        # Tilt the planet according to a given obliquity at initialization 
        # Obliquity will be given in degrees but is changed to radians when making the rotation matrix
        rotMatrix = makeRotationMatrix(obliq, 'y') #create a rotation matrix that will rotate the mesh obliq degrees about x axis
        self.mesh.apply_transform(rotMatrix)
        if not start:
            self.update()
            
    def rotateWithObliquity(self,obliq, rot):
        # obliq: obliquity tilt in deg.
        # rot: diurnal rotation amount (about polar axis) in deg. 
        # Assumes initial obliquity shift has already been applied 
        
        # define rotation matrices
        mat1 = makeRotationMatrix(-obliq, 'y')#untilt
        mat2 = makeRotationMatrix(rot, 'z') #rotate
        mat3 = makeRotationMatrix(obliq, 'y') #tilt
        
        # apply matrices to mesh
        self.mesh.apply_transform(mat1) #undo tilt
        self.mesh.apply_transform(mat2) # rotate
        self.mesh.apply_transform(mat3) # tilt 
        
        # update shape model parameters 
        self.update()
        
    
def makeRotationMatrix(theta, axis):
    # Creates a 4x4 rotation matrix that can be applied to a mesh
    # Inputs are a rotation angle theta (in degrees) and the axis you want the rotation to be around
    theta = np.radians(theta)
    if axis == 'x':
        rotMatrix = np.asarray([1,0,0,0,0,np.cos(theta),np.sin(theta),0,0,-np.sin(theta),np.cos(theta),0,0,0,0,1]).reshape((4,4))
    elif axis == 'y':
        rotMatrix = np.asarray([np.cos(theta),0,- np.sin(theta),0,0,1,0,0,np.sin(theta),0,np.cos(theta),0,0,0,0,1]).reshape((4,4))
    elif axis == 'z':
        rotMatrix = np.asarray([np.cos(theta),-np.sin(theta),0,0,np.sin(theta),np.cos(theta),0,0,0,0,1,0,0,0,0,1]).reshape((4,4))
    return rotMatrix

def fluxIntensity(solar,rDist,s,n):
    # Called from FluxCalc
    flux = (solar / (rDist**2)) * np.dot(s,n)
    return flux

def normalize(n):
    # After finding the normal vector for a face, normalize it to length 1
    length = math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
    normal = n / length
    return normal

'''
def rotate(n,theta):
    #print(n)
    nx,ny,nz = n[0],n[1],n[2]
    newX = np.cos(theta)*nx - np.sin(theta)*ny
    newY = np.sin(theta)*nx + np.cos(theta)*ny
    newN = [newX,newY,nz]
    return newN 
'''

# the rotate function above is the original. This has the z part changed to try to fit with the geometry of the bowl shaped crater
def rotate(n,theta):
    #print(n)
    nx,ny,nz = n[0],n[1],n[2]
    newX = np.cos(theta)*nx - np.sin(theta)*nz
    newZ = np.sin(theta)*nx + np.cos(theta)*nz
    newN = [newX,ny,newZ]
    return newN    
    



