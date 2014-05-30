# -*- coding: utf-8 -*-
"""
Created on Fri May 30 08:45:21 2014

@author: manuamador@gmail.com
"""
from __future__ import division
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#scikit image modules
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb

im = Image.open("test.jpg")  #loading the image
image_rvb = np.asarray(im, dtype=np.uint8)  #convert the image to a numpy array
image = image_rvb[:,:,0] #analyzing the red channel
max_y=len(image[:,0]) #extracting the size of the image
max_x=len(image[0,:])

# apply threshold to remove the background and the noise
thresh = threshold_otsu(image)
bw = closing(image < thresh, square(10)) #closing the objects in the images to get the mask of the objects

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)

# label image regions
label_image = label(cleared)
borders = np.logical_xor(bw, cleared)
label_image[borders] = -1
image_label_overlay = label2rgb(label_image, image=image)

def traitement(minr,minc,maxr,maxc):
    """
    Assign a number to the objects as long as they are placed on a 3x4 array
    """
    if minc/max_x<0.2:
        if minr/max_y<0.33:
            s=0
        if minr/max_y<0.66 and minr/max_y>0.33:
            s=4
        if minr/max_y>0.66:
            s=8
    if minc/max_x>0.2 and minc/max_x<0.4:
        if minr/max_y<0.33:
            s=1
        if minr/max_y<0.66 and minr/max_y>0.33:
            s=5
        if minr/max_y>0.66:
            s=9
    if minc/max_x>0.4 and minc/max_x<0.6:
        if minr/max_y<0.33:
            s=2
        if minr/max_y<0.66 and minr/max_y>0.33:
            s=6
        if minr/max_y>0.66:
            s=10
    if minc/max_x>0.6 :
        if minr/max_y<0.33:
            s=3
        if minr/max_y<0.66 and minr/max_y>0.33:
            s=7
        if minr/max_y>0.66:
            s=11
    return s

#relative positions of the leds from the upper left corner
offsetled0=np.array([46,88])
offsetled1=np.array([55,88])

#half length of the square area
dx=4
dy=4

limit=150 #decision threshold level
def OnorOff(image,minc,minr,offsetled0,offsetled1,lim=limit):

    if np.mean(image[minr+offsetled0[0]-dx:minr+offsetled0[0]+dx,minc+offsetled0[1]-dy:minc+offsetled0[1]+dy])>lim:
        led0=1
    else:
        led0=0
    if np.mean(image[minr+offsetled1[0]-dx:minr+offsetled1[0]+dx,minc+offsetled1[1]-dy:minc+offsetled1[1]+dy])>lim:
        led1=1
    else:
        led1=0
    return np.array((led0,led1))


#plot and leds status analysis
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 6))
ax.imshow(image_label_overlay)

result=np.zeros((12,2))
for region in regionprops(label_image):
    # skip small images
    if region.area < 100:
        continue
    minr, minc, maxr, maxc = region.bbox
    s=traitement(minr,minc,maxr,maxc)
    result[s,:]=OnorOff(image,minc,minr,offsetled0,offsetled1,limit)
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(minc+0.5*(maxc-minc), minr+0.5*(maxr-minr), str(s), size=50,color='r')
    ax.plot(minc+offsetled0[1],minr+offsetled0[0],'k+')
    ax.plot(minc+offsetled1[1],minr+offsetled1[0],'ks')
print result
plt.show()
