#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:58:59 2021

@author: kyso3185

3-d k-d tree code
"""


"""
3-d k-d tree code
Builds from version found in kdtree_2d_Kya.py, 
Which borrows some functions from: 
https://www.astroml.org/book_figures/chapter2/fig_kdtree_example.html
"""

import numpy as np
from matplotlib import pyplot as plt
import weakref

# global
global hit 

class KDTree:
    
    def __init__(self, obj, leaf_size=5, max_depth=10):
        
        # Number of objects
        n = len(obj)  #Could be replaced with "n" fed in 
        
        # Data array contains center coords. of primitives in the tree
        data = np.zeros([n, 3])
        for i in np.arange(0, n):
            data[i,:] = obj[i].centroid
        self.data = np.asarray(data)
        
        # 'leaf_size' is number of primitives in the leaves
        self.leaf_size = leaf_size
        
        # 'max_depth' is the 
        self.max_depth = max_depth

        # Find bounding box for full dataset
        if (obj[0].xyzmin is not None):
            xyzmin = np.zeros([n, 3])
            xyzmax = np.zeros([n, 3])
            for i in np.arange(0, n):
                xyzmin[i,:] = obj[i].xyzmin
                xyzmax[i,:] = obj[i].xyzmax
            mins = xyzmin.min(0)
            maxs = xyzmax.max(0)
        else:
            mins = data.min(0)
            maxs = data.max(0)
            
        
        # Create bounding box
        self.bbox = bbox(mins, maxs)
        
        # Root is at depth = 0
        self.depth = 0
        
        # Create root node and recursively create all branches
        self.root = KDNode(self, obj, self.bbox.mins, self.bbox.maxs, 
                           leaf_size, max_depth, obj)
        
        
    def ShadowCheck(self,ray,expTri,baseDist):
        # Shadowing algorithm that uses the KD Tree structure
        # Starts from the root and uses the same intersection algorithm as 
        #   used in the brute force method, just casts fewer rays ideally
        # Returns True if facet is shadowed and False if no shadowing detected
        shadow = self.root.Shadow_Traversal(ray = ray,expTri = expTri,baseDist = baseDist)
        
        return shadow
    
        
            

class KDNode:
    """
    Simple KD-tree node class
    
    Args:
        data = [Nx2] array of data points
        mins = [1x2] array of minimum values in each dimension
        maxs = [1x2] array of maximum values in each dimension
    """
    
    # Initialize the class
    def __init__(self, parent, obj, 
                 mins, maxs, leaf_size, max_depth,assoc_obj):
        self.obj_3 = []
        self.obj_4 = []
        self.assoc_obj = assoc_obj
        
        # Parent of current node
        self.parent = weakref.ref(parent)
        
        # Depth of this node in the tree
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        #print ("Depth: "+ str(self.depth))
        # Number of objects
        n = len(obj)     # REDUNDANT. Again, could spead this up by passing in 
        #print ("Depth: "+ str(self.depth) + " and n: "+str(n))
        
        # Find minima and maxima of vertices
        xyz = np.zeros([n, 3])
        for i in np.arange(0, n):
            xyz[i,:] = obj[i].centroid
        self.xyz = np.asarray(xyz)
        
        # Check that data points are 3-d
        # First dimension spans 'rows' of data points
        assert self.xyz.shape[1] == 3
        
        self.bbox = bbox(mins, maxs)
        
        self.child1 = None
        self.child2 = None
        
        # Check that we haven't reached the leaf, then subdivide
        if n > leaf_size and self.depth <= max_depth:
            self.isLeaf = False
            
            # We want to subdivide on the largest axis
            # First step is to sort data along this dimension
            split_axis = np.argmax(self.bbox.sizes)
            self.split_axis = split_axis
            self.obj = sorted(obj, 
                              key=lambda triangle: triangle.centroid[split_axis])    #what was rect.xy? 
            
            # Number of objects
            n = len(self.obj)
            xyz = np.zeros([n, 3])
            for i in np.arange(0, n):
                xyz[i,:] = self.obj[i].centroid ####################################
            self.xyz = np.asarray(xyz)
            
            # Find the split point at the median
            half_n = n // 2
            # Take the mid-point between two points to separate them
            split_point = 0.5 * (self.xyz[half_n, split_axis]
                                 + self.xyz[half_n - 1, split_axis])
            self.split_point = split_point
            #print (split_axis)
            #print ("Split point: "+str(split_point))
            
            
            # Bounding box defines child nodes
            mins1 = self.bbox.mins.copy()
            mins1[split_axis] = split_point
            maxs2 = self.bbox.maxs.copy()
            maxs2[split_axis] = split_point
            
            ####################### FIX THE SPLIT ############################
            # Build self.obj_3 and self.obj_4
            # check if any part of triangle is on the side of the split point that we're considering 
            # then objects in recursive sub nodes are self.obj_3
            
            for thing in self.assoc_obj: #for each triangle child has recieved 
                # check which side of split point
                # remember only doing this for the split axis you're currently working with
                # if lower, send to obj_3, if higher, obj_4. If both, add to both
                # use xyzmin and yxzmax
                smaller = thing.xyzmin[split_axis] <= split_point
                larger = thing.xyzmax[split_axis] >= split_point
                if smaller and larger:
                    self.obj_3.append(thing)
                    self.obj_4.append(thing)
                elif smaller:
                    self.obj_3.append(thing)
                elif larger: 
                    self.obj_4.append(thing)
                else: 
                    print ("ERROR: Splitting Failed")
            
            
            # Recursively build sub-nodes
            self.child1 = KDNode(self, self.obj[half_n:], 
                                 mins1, self.bbox.maxs, 
                                 leaf_size, max_depth, self.obj_3)
            self.child2 = KDNode(self, self.obj[:half_n], 
                                 self.bbox.mins, maxs2,  
                                 leaf_size, max_depth, self.obj_4)
            
            
        else:
            self.isLeaf = True
            self.split_point = None
            #self.obj = obj
            self.split_axis = None
            
            
    def draw_triangle(self, ax, depth=0):
            """
            Recursively draw bounding boxes for each node in the tree
            """
            if depth == 0:
                tri = plt.triangle(self.bbox.mins, *self.bbox.sizes, 
                                     ec='k', fc='none')
                ax.add_patch(tri)
            
            # Check that children exist, then do recursion
            if self.child1 is not None and depth > 0:
                self.child1.draw_triangle(ax, depth-1)
                self.child2.draw_triangle(ax, depth-1)

    
    
    def Traversal_old(self,ray=None, hitObj = None, count = 0, colors = None, epsilon = 0.01):#, checked_tris = []):
        # test for ray-triangle intersection 
        hit = False
        hitcount = 0
        
        if self.isLeaf:
            for tri in self.assoc_obj:
                count += 1
                colors[tri.num] = 0.5
                if tri.intersect_simple(ray):
                    colors[tri.num] = 0.99
                    hitcount += 1
                    hit = True
                    print ("KD TREE INTERSECT DETECTED")
                    print ("Number: "+str(tri.num))
                    if hitcount <= 1:                               #maybe sort instead. Also maybe sort objects when defining node
                        dist = find_distance(ray=ray, tri = tri)
                        closest_tri = tri
                    else:
                        dist2 = find_distance(ray = ray, tri = tri)
                        if dist2 < dist:
                            print ("Closer triangle found")
                            dist = dist2
                            closest_tri = tri
            if hit: 
                #objects.append(closest_tri)
                hitObj = closest_tri

                
        # traverse children nodes 
        if not hit and not self.isLeaf:
            if ray.o[self.split_axis] <= self.split_point:
                ### recursively traverse left node
                hit, count, hitObj, colors = self.child1.Traversal(ray, hitObj,count = count,colors = colors)
                ### if not hit and ray.direction[axis] >= -0.01 
                ### (if no intersections found when traversing  left node and   
                ### ray direction is positive. Traverse right)
            
                if not hit and ray.d[self.split_axis] >= -epsilon:
                    #### recursively traverse right node 
                    hit, count, hitObj, colors = self.child2.Traversal(ray, hitObj,count = count,colors = colors)
            
            
            else:
                ### Recursively traverse right node 
                ### (if position  greater than split plane )
                hit, count, hitObj, colors = self.child2.Traversal(ray,hitObj, count = count,colors = colors )
                ### if not hit and ray.direction[axis] < 0.01
                if not hit and ray.d[self.split_axis] <= epsilon:
                    #### recursively traverse left node 
                    hit, count, hitObj, colors = self.child1.Traversal(ray, hitObj,count = count,colors = colors )
        
        return hit, count, hitObj,colors#, checked_tris
    
    
    def Traversal(self,ray=None, hitObj = None, count = 0, colors = None, epsilon = 0.01):#, checked_tris = []):
        # test for ray-triangle intersection 
        # Includes consideration of whether a ray starts from a given triangle (this will be necessary when reflection is implemented)
        hit = False
        hitcount = 0
        
        if self.isLeaf:
            for tri in self.assoc_obj:
                count += 1
                colors[tri.num] = 0.5
                if tri.intersect_simple(ray)==1:
                    colors[tri.num] = 0.99
                    hitcount += 1
                    hit = True
                    print ("KD TREE INTERSECT DETECTED")
                    print ("Number: "+str(tri.num))
                    if hitcount <= 1:                               #maybe sort instead. Also maybe sort objects when defining node
                        dist = find_distance(ray=ray, tri = tri)
                        closest_tri = tri
                    else:
                        dist2 = find_distance(ray = ray, tri = tri)
                        if dist2 < dist:
                            print ("Closer triangle found")
                            dist = dist2
                            closest_tri = tri
            if hit: 
                hitObj = closest_tri

                
        # traverse children nodes 
        if not hit and not self.isLeaf:
            ## if ray.position[axis] <= split_plane (ray begins left of split 
            ## plane. Check left first)
            if ray.o[self.split_axis] <= self.split_point:
                ### recursively traverse left node
                hit, count, hitObj, colors = self.child1.Traversal(ray, hitObj,count = count,colors = colors)
                ### if not hit and ray.direction[axis] >= -0.01 
                ### (if no intersections found when traversing  left node and   
                ### ray direction is positive. Traverse right)
            
                if not hit and ray.d[self.split_axis] >= -epsilon:
                    #### recursively traverse right node 
                    hit, count, hitObj, colors = self.child2.Traversal(ray, hitObj,count = count,colors = colors)
            
            
            else:
                # Recursively traverse right node 
                # (if position  greater than split plane )
                hit, count, hitObj, colors = self.child2.Traversal(ray,hitObj, count = count,colors = colors )
                # if not hit and ray.direction[axis] < 0.01
                if not hit and ray.d[self.split_axis] <= epsilon:
                    # ecursively traverse left node 
                    hit, count, hitObj, colors = self.child1.Traversal(ray, hitObj,count = count,colors = colors )
        
        return hit, count, hitObj,colors
    
    
    
    
    
    def Initial_Traversal(self, tris, ray, expTri,orgTri):
        # Alternative to KD tree. Checks all tris for intersection and returns the closest
        distExp = find_distance(tri = expTri,ray = ray)
        hitcount = 0
        
        #print ("Checking")
        for tri in tris: 
            if tri == orgTri:
                #print ("Origin tri")
                continue
            if tri == expTri:
                #print ("Expected tri")
                continue
            if tri.intersect_simple(ray) == 1:
                hitcount += 1
                #hit = True
                if hitcount <= 1:
                    dist = find_distance(ray=ray,tri = tri)
                    #closest_tri = tri
                
                else:
                    dist2 = find_distance(ray=ray,tri=tri)
                    if dist2 < dist:
                        dist = dist2
                if dist < distExp:
                        return False
        
        return True
    
    
    
    
    
    
    def Initial_Traversal_KD(self,ray=None, hitObj = None, count = 0, expTri = None,epsilon = 0.01):#, checked_tris = []):
        # test for ray-triangle intersection 
        # Includes consideration of whether a ray starts from a given triangle (this will be necessary when reflection is implemented)
        hit = False
        hitcount = 0
        
        if self.isLeaf:
            distExp = find_distance(ray = ray,tri = expTri)
            for tri in self.assoc_obj:
                count += 1
                if tri.intersect_simple(ray)==1:
                    hitcount += 1
                    hit = True
                    if hitcount <= 1:                         
                        dist = find_distance(ray=ray, tri = tri)
                        closest_tri = tri
                    else:
                        dist2 = find_distance(ray = ray, tri = tri)
                        if dist2 < dist:
                            dist = dist2
                            closest_tri = tri
                    if dist > distExp:
                        print ("Intersect is farther than expected triangle")
            if hit: 
                hitObj = closest_tri

                
        # traverse children nodes 
        if not hit and not self.isLeaf:
            # If ray.position[axis] <= split_plane (ray begins left of split plane. Check left first)
            if ray.o[self.split_axis] <= self.split_point:
                # Recursively traverse left node
                hit = self.child1.Initial_Traversal(ray, hitObj,count = count,expTri = expTri)
                # If not hit and ray.direction[axis] >= -0.01 
                # (if no intersections found when traversing  left node and   
                # Ray direction is positive. Traverse right)
            
                if not hit and ray.d[self.split_axis] >= -epsilon:
                    # Recursively traverse right node 
                    hit = self.child2.Initial_Traversal(ray, hitObj,count = count,expTri = expTri)
            
            
            else:
                # Recursively traverse right node 
                # (if position  greater than split plane )
                hit = self.child2.Initial_Traversal(ray,hitObj, count = count,expTri = expTri)
                # If not hit and ray.direction[axis] < 0.01
                if not hit and ray.d[self.split_axis] <= epsilon:
                    # Recursively traverse left node 
                    hit = self.child1.Initial_Traversal(ray, hitObj,count = count,expTri = expTri)
                    

        if hit: 
            if hitObj == expTri:
                return True
        return False

    
    
    
    
    def Shadow_Traversal(self,ray=None, hitObj = None, expTri = None,baseDist = None, epsilon = 0.01): #, checked_tris = []):
        # test for ray-triangle intersection 
        # Includes consideration of whether a ray starts from a given triangle (this will be necessary when reflection is implemented)
        hit = False                                        
            
        if self.isLeaf: # Reached leaves of tree
            for tri in self.assoc_obj: # Check tris associated with leaf
                result = tri.intersect_simple(ray)
                if result == 1: # Intersect detected
                    if tri.num == expTri.num: # If this is the triangle we want to hit
                        # Skip to next tri
                        hitObj = expTri
                        continue
                    dist = find_distance(ray = ray, tri = tri)
                    if dist < baseDist:
                        # closer intersection found. Light is stopped before reaching facet 
                        # shadowed 
                        hit = True
                        hitObj = tri
                        return True 
                    elif dist == baseDist:
                        if tri != expTri:
                            print ("Diff Tri, same distance")
                            continue
                    else: # Intersects, but farther than expected triangle and thus no shadowing. Skip to next
                        print ("Intersected tri is farther away than expected No. "+str(expTri.num))
                        continue
                if result == 2: 
                    print ("Something's gone wrong. Ray starting point is in triangle")
                    
            hit = False # Light makes it to expected facet. No shadowing detected in leaf objects 


                
        # traverse children nodes 
        if not hit and not self.isLeaf:
            ## if ray.position[axis] <= split_plane (ray begins left of split plane. Check left first)
            if ray.o[self.split_axis] <= self.split_point:
                ### recursively traverse left node
                hit = self.child1.Shadow_Traversal(ray, hitObj,expTri = expTri)
                ### if not hit and ray.direction[axis] >= -0.01 
                ### (if no intersections found when traversing  left node and   
                ### ray direction is positive. Traverse right)
            
                if not hit and ray.d[self.split_axis] >= -epsilon:
                    #print ("ray direction is positive")
                    #### recursively traverse right node 
                    hit = self.child2.Shadow_Traversal(ray, hitObj,expTri = expTri)
            
            
            else:
                #print ("more than origin")
                ### Recursively traverse right node 
                ### (if position  greater than split plane )
                hit = self.child2.Shadow_Traversal(ray,hitObj,expTri = expTri)
                ### if not hit and ray.direction[axis] < 0.01
                if not hit and ray.d[self.split_axis] <= epsilon:
                    #print ("ray direction is negative")
                    #### recursively traverse left node 
                    hit = self.child1.Shadow_Traversal(ray, hitObj,expTri = expTri)
                    

        if hit: 
            # Intersection found in tree objects  
            if hitObj == expTri: # Expected tri is intersected: no shadowing 
                # This is redundant since I include expected triangle check in the intersection algorithm. Should delete 
                return False
            return True # facet was hit and it wasn't expected triangle. Shadowed. 
        else:
            return False # No hit detected. Recursively return false and move to next leaf  


    
def find_distance(ray = None, tri = None):
    trix, triy, triz = tri.centroid[0],tri.centroid[1],tri.centroid[2]
    dist = np.sqrt((trix - ray.o[0])**2 + (triy - ray.o[1])**2 + (triz - ray.o[2])**2)
    return dist

                
        
            
class bbox:
    """
    A bounding box with given ranges in (x,y) space
    """
    def __init__(self, mins, maxs):
        self.xmin = mins[0]
        self.xmax = maxs[0]
        self.ymin = mins[1]
        self.ymax = maxs[1]
        self.zmin = mins[2]
        self.zmax = maxs[2]
        self.sizes = maxs - mins
        self.mins = mins
        self.maxs = maxs


    def intersect(self, ray):
        # won't be using this 
        tminx = (self.xmin - ray.o[0])/ray.d[0]
        tminy = (self.ymin - ray.o[1])/ray.d[1]
        tmaxx = (self.xmax - ray.o[0])/ray.d[0]
        tmaxy = (self.ymax - ray.o[1])/ray.d[1]
        
        if (tminx>tmaxy) or (tminy>tmaxx):
            t = []
        else:
            tmin = np.min([tminx,tminy])
            tmax = np.max([tmaxx,tmaxy])
            t = np.asarray([tmin, tmax])
        
        return(t)
    
    def intersectBoolean(self, ray):
        # will probably delete this
        tminx = (self.xmin - ray.o[0])/ray.d[0]
        tminy = (self.ymin - ray.o[1])/ray.d[1]
        tmaxx = (self.xmax - ray.o[0])/ray.d[0]
        tmaxy = (self.ymax - ray.o[1])/ray.d[1]
        
        if (tminx>tmaxy) or (tminy>tmaxx):
            return False
        else:
            return True
            