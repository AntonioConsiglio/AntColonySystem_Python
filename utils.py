from node import Node
import cv2
import numpy as np
import random
import os
from datetime import datetime
from shutil import copy
import config

EXP_PATH = "./experimets"
global LAST_CREATED_PATH
LAST_CREATED_PATH = None

def save_experiment_result(image,iter):
    global LAST_CREATED_PATH
    if LAST_CREATED_PATH is None:
        LAST_CREATED_PATH = os.path.join(EXP_PATH,datetime.today().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(LAST_CREATED_PATH,exist_ok=True)
    filename = os.path.join(LAST_CREATED_PATH,f"plot_{iter}.png")
    cv2.imwrite(filename,(image*255).astype(int))
    copy(os.path.join(os.getcwd(),"config.py"),os.path.join(LAST_CREATED_PATH,"config.py"))


def generate_node_map(n:int,seed=42):

    random.seed(seed)
    node_map = []
    for _ in range(n):
        node = Node()
        node_map.append(node)
    
    random.seed(None)
    return node_map


def plot_cv_map(node_map,Hmap=101,Wmap=101):
    h = Hmap+50
    w = Wmap
    imgmap = np.ones((h,w,3))
    for node in node_map:
        cv2.circle(imgmap,node.pos,config.CIRCLE_RADIUS,(255,0,0),-1)
    contours = np.array( [ [0,w], [w,w], [w,h], [0,h] ] )
    imgmap = cv2.drawContours(imgmap, [contours], -1, color=(0, 0, 0), thickness=cv2.FILLED)
    return imgmap

def generate_final_result(figure,image):
    final_image = np.hstack([figure/255.0,image])
    return final_image

def draw_connection(result_map,imagemap,mid_distance=None,best_iteration=None):
    start = False
    node = result_map.result
    # for node in nodemap:
    while node.child is not None:
        if not start:
            cv2.line(imagemap,node.pos,node.child.pos,(255,0,0),2)
            start = True
        elif node.child.child is None:
            cv2.line(imagemap,node.pos,node.child.pos,(0,255,0),2)
        else:
            cv2.line(imagemap,node.pos,node.child.pos,(0,0,0),1)
        node = node.child

    if mid_distance is not None:
        imagemap = add_text_result(imagemap,f"Min Distance: {mid_distance:.1f}",0.1)
    if best_iteration is not None:
        imagemap = add_text_result(imagemap,f"Best Iter: {best_iteration}",0.6)

    return imagemap

def draw_pheromone(nodes,imagemap,pheromone_matrix:np.ndarray):
    
    max_pheromone = pheromone_matrix.max()

    image = imagemap.copy()
    for n,node in enumerate(nodes):
        for np, child_node in  enumerate(nodes[n+1:],start=n+1):
            p_value = pheromone_matrix[n,np]
            linesize = (20/max_pheromone**1.3) * p_value**1.3
            if linesize < 1: continue
            cv2.line(image,node.pos,child_node.pos,(0,0,50),int(linesize))
    imagemap = cv2.addWeighted(image,0.2,imagemap,0.8,0)
    return imagemap
    
def add_text_result(imagemap,text,pos_ratio):
    h,w,_ = imagemap.shape
    y_text = h - int((h-w)*0.4)
    x_test = int(w*pos_ratio)
    cv2.putText(imagemap,text,(x_test,y_text),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
    
    return imagemap

import time

def average_runtime(count=10):

    def decorator(func):
        counter = 0
        total_time = 0
        
        def wrapper(*args, **kwargs):
            nonlocal counter, total_time
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            counter += 1
            total_time += end_time - start_time
            
            if counter % count == 0:
                avg_time = total_time / count
                print(f"\nAverage runtime of {func.__name__}: {avg_time*1000:.2f} ms over last {count} calls")
                total_time = 0
            
            return result
        
        return wrapper
    
    return decorator


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


