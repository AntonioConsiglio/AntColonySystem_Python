# generate an initial candidate

# Get an initial temperauure

# Iterate for a Numnber of Iteration
# Sample from a symmetrical distibution (Normal distrinbuton)
# add the sampled value to the initial candidate
# calculate a probability p equal to exp(deltaH/T)  h is the euristic function
# Accept the new x if the probability of x' (p) is greater than u
# Update the temperature with and alpha decay (0-1)
from node import Node
import cv2
import numpy as np
import random

def generate_node_map(n:int,seed=42):

    random.seed(seed)
    node_map = []
    for _ in range(n):
        node = Node()
        node_map.append(node)
    
    random.seed(None)
    return node_map


def plot_cv_map(node_map,Hmap=101,Wmap=101):

    imgmap = np.ones((Hmap+50,Wmap,3))
    for node in node_map:
        cv2.circle(imgmap,node.pos,3,(255,0,0),-1)
    return imgmap

def generate_final_result(figure,image):
    final_image = np.hstack([figure/255.0,image])
    return final_image

def draw_connection(result_map,imagemap,mid_distance=None):
    start = False
    node = result_map.result
    # for node in nodemap:
    while node.child is not None:
        if not start:
            cv2.line(imagemap,node.pos,node.child.pos,(255,0,0),1)
            start = True
        elif node.child.child is None:
            cv2.line(imagemap,node.pos,node.child.pos,(0,255,0),1)
        else:
            cv2.line(imagemap,node.pos,node.child.pos,(0,0,255),1)
        node = node.child
    if mid_distance is not None:
        imagemap = add_min_distance_result(imagemap,mid_distance)
    return imagemap

def add_min_distance_result(imagemap,distance):
    h,w,_ = imagemap.shape
    y_text = h - int((h-w)*0.4)
    x_test = int(w*0.2)
    contours = np.array( [ [0,w], [w,w], [w,h], [0,h] ] )
    cv2.drawContours(imagemap, [contours], -1, color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(imagemap,f"min Disance: {distance:.1f}",(x_test,y_text),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)
    
    return imagemap

if __name__ == "__main__":

    nodemap = generate_node_map(20)
    imagmap = plot_cv_map(nodemap,400,400)
    for n,node in enumerate(nodemap,start=1):
        print(n)
        if n == len(nodemap):continue
        node.child = nodemap[n]

    imagmap = draw_connection(nodemap,imagmap)
    cv2.imshow("Image Map",imagmap)
    cv2.waitKey(0)