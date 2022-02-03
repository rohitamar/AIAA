import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import cv2
import math
import sys 

#parameters
n_clusters = 5

#helper method for BFS
#returns nearest color
def isValid():
    return False

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

#Breadth First Search for grids
#note that tuples are being used for grid
#node: (r,c)
def BFS(grid, vis, r, c):
    q = queue()

    q.append((r, c))
    vis[r][c] = True

    while(q.empty() == False):
        curr = q.popLeft()
        for i in range(4):
            adjx = curr[0] + dx[i]
            adjy = curr[1] + dy[i]
            if(isValid(adjx, adjy)):
                q.append((adjx, adjy))
                vis[adjx][adjy] = True

def distRGB(rgb1, rgb2):
    r = rgb1[0] - rgb2[0]
    g = rgb1[1] - rgb2[1]
    b = rgb1[2] - rgb2[2]
    return math.sqrt(r*r + g*g + b*b)

colorCounter = [0] * n_clusters

img = cv2.imread("1.jpg", cv2.IMREAD_COLOR)
#given a rgb grid, finds the number of each color (from kmeans)
def find_color_distribution():
    m = sys.maxsize
    argmin = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            m = sys.maxsize
            for k in range(4):
                distance = distRGB(img[i,j], colors[k])
                if distance < m:
                    argmin = k
                    m = distance
            colorCounter[argmin] = colorCounter[argmin] + 1

    plt.imshow(img)
    plt.show()

    print(colors)
    print(colorCounter)

    
ori = img.reshape((img.shape[0] * img.shape[1], 3))
kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(ori)
colors = kmeans.cluster_centers_
find_color_distribution()
#make visited array for the grid
vis = [[False for i in range(img.shape[0])] for j in range(img.shape[1])]
#BFS(img, vis, 0, 0)

n_clusters
