#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:20:58 2021

@author: kyso3185
"""


import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d

class small_body:
    def __init__(self, name, filePath):
        self.name = name

    


class ray:

    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction
        self.vec = self.d
    
    def draw(self, ax, length, c = "r",label = None):
        '''
        Plot the ray using given length
        '''
        o = self.o
        d = self.d
        
        p = o + length * np.asarray(d/np.linalg.norm(d))
        x = np.asarray([o[0],p[0]])
        y = np.asarray([o[1],p[1]])
        z = np.asarray([o[2],p[2]])
        if label == None:
            l = art3d.Line3D(x,y,z,c=c)
        else: 
            l = art3d.Line3D(x,y,z,c=c,label = label)
        ax.add_line(l)
        
    def normalize(self, n):
        # After finding the normal vector for a face, normalize it to length 1
        length = np.linalg.norm(n)
        normal = n / length
        return normal

class triangle:
    def __init__(self,v1,v2,v3,normal,num=0): #u=0,v=0 removed
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.side1 = np.asarray(self.v2 - self.v1)
        self.side2 = np.asarray(self.v3 - self.v1)
        self.normal = normal
        self.num = num #which tri are we talking about (for color array) 
        
        self.verts = np.asarray([v1,v2,v3])
        
        self.xyzmin = np.min(self.verts,0)
        self.xyzmax = np.max(self.verts,0)
            
        self.centroid = np.mean(self.verts,axis = 0)
            
        self.area = self.findArea()
        self.r, self.long, self.lat = self.cart2latlong()
        self.rVect = self.findRVect()

    
    def findArea(self):
        cross = np.cross(self.side1,self.side2)
        mag = np.linalg.norm(cross)
        area = 0.5 * mag
        return area
    
    def cart2latlong(self):
        r = np.sqrt(self.centroid[0]**2 + self.centroid[1]**2 + self.centroid[2]**2)
        long = np.arctan2(self.centroid[1], self.centroid[0])*(180 / np.pi)
        if long < 0:
            long = 360.+long
        lat = np.arcsin(self.centroid[2] / r)*(180/np.pi)
        return r, long,lat
    
    def findRVect(self):
        # Find the radial vector from the origin to the centroid of the facet
        rVect = np.asarray(self.centroid-np.asarray([0,0,0])) 
        return rVect / np.linalg.norm(rVect)
                
        
    def intersect_simple(self, ray):
    # Inspired in part by code from Erik Rotteveel
    # returns 0.0 if no intersect
    # returns 1.0 if intersect
    # returns 2.0 if starting point is in triangle (shouldn't be an issue for now)
        side1 = self.side1
        side2 = self.side2
        normal = self.normal
        
        b = np.inner(normal,ray.vec) #checks if ray is parallel to plane of triangle
        a = np.inner(normal,np.asarray(self.v1 - ray.o)) #checks if ray is outside triangle
        
        if b == 0.0:#ray parallel to plane
            if a != 0.0:#ray is outside triangle but still parallel to plane
                return 0 
            else: 
                # ip is intersect point
                ip = 0.0 #Ray is parallel and lies in plane
        
        
        else: # Ray is not parallel to plane of triangle?
            ip = a/b
        
        if ip < 0.0:
            return 0
        
        w = np.asarray(ray.o + ip * (ray.vec) - self.v1)
            
        denom = np.inner(side1,side2) * np.inner(side1,side2) - np.inner(side1,side1) * np.inner(side2,side2)
        si = (np.inner(side1,side2) * np.inner(w,side2) - np.inner(side2,side2) * np.inner(w,side1)) / denom
            
        if (si < 0.0) | (si > 1.0):
            return 0
            
        ti = (np.inner(side1,side2) * np.inner(w,side1) - np.inner(side1,side1) * np.inner(w,side2)) / denom
            
        if (ti < 0.0) | (si + ti > 1.0):
            return 0
        
        if (ip == 0.0):
            # point 0 lies on the triangle
            # If checking for point inside polygon, return 2 so that the lopo over triangles stops
            return 2
        
        return 1 #if all other conditions fail, intersect
    
    
    
      
        
        
        
class facet: 
    def __init__(self, xyz, side, num=0):
        self.xyz = xyz   # locations of centroid
        self.num = num # Identifying number for facet 
        
        height = np.sqrt(3) / 2 * side
        dist_to_vert = (2 / 3) * height
        opp_dist = (1 / 3) * height 
        
        # define separate vertices
        self.v1 = [xyz[0], xyz[1], xyz[2] + dist_to_vert]
        self.v2 = [xyz[0] - side/2, xyz[1], xyz[2] - opp_dist]
        self.v3 = [xyz[0] + side/2, xyz[1], xyz[2] - opp_dist]
        
        # self.v1, self.v2, self.v3 = xy[0], xy[1], xy[2]
        
        # find min and max values for each coordinate 
        self.xmin = np.min(xyz,0)
        self.xmax = np.max(xyz,0) 
        
        self.ymin = np.min(xyz,1)
        self.ymax = np.max(xyz,1)
        
        self.zmin = np.min(xyz,2)
        self.zmax = np.max(xyz,2)
        
        
        
    def intersection(self,ray):
        m = ray.d[1]/ray.d[0] # slope of ray
        b = ray.o[1] - m * ray.o[0] # y-intercept
        
        # Vertex coords.
        x = self.xyz[:,0]
        y = self.xyz[:,1]
        z = self.xyz[:,2]
        
        # Vertex coords. relative to span of ray
        s1 = np.sign(m*x[0] - y[0] + b)
        s2 = np.sign(m*x[1] - y[1] + b)
        s3 = np.sign(m*x[2] - y[2] + b)
        s4 = np.sign(m*x[3] - y[3] + b)
        
        
    
class body:
    def __init__(self,faces,verts,norms):
        self.faces = faces
        self.verts = verts
        self.normals = norms
        