from abc import ABC,abstractmethod
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import heapq
import random
from copy import deepcopy
from node import Node
from typing import Optional
import config

import matplotlib.pyplot as plt

class PathResult():
    def __init__(self,):
        self._total_distance = 0
        self.result = None
    
    @property
    def total_distance(self,):
        return self._total_distance
    
    def add_child(self,node):
        if self.result is None:
            self.result = node
        else:
            # print(f"{self.total_distance=}")
            self._total_distance += self.result.get_distance(node)
            node.child = self.result
            self.result = node
            # print(self.result)


@dataclass
class ACS(ABC):
    alpha:float
    beta:float
    max_iter:int
    n_ants:int
    random_state:Optional[int]
    
    def __post_init__(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    @abstractmethod
    def calculate_path(self,graph):
        return NotImplementedError
    
    @abstractmethod
    def single_ant_path(self,ant,graph):
        return NotImplementedError

    def single_iteration(self,graph):
        return NotImplementedError

    @abstractmethod
    def update_pheromone(self,):
        return NotImplementedError
        
    

class BaseAntColonySystem(ACS):
    def __init__(self,alpha,beta,max_iter,n_ants,Q=1,p_decay=0.2,random_state=None):
        super().__init__(alpha,beta,max_iter,n_ants,random_state)
        self.Q = Q
        self.p_decay = p_decay
        self.path_distance_result = []

    def calculate_path(self, graph):
        self.graph_index = {obj: i for i, obj in enumerate(graph)}
        self.results = []
        self.adj_pheromone_matrix = self.initialize_adj_p_matrix(graph)
        iter_loop = tqdm(range(self.max_iter))
        for j in iter_loop:
            iter_loop.set_description(desc=f"Iter {j}: ")
            distance,_,path = self.single_iteration(graph)
            iter_loop.set_postfix(curr_min_distance = distance)
            self.path_distance_result.append(distance)
            heapq.heappush(self.results,(distance,j,path))

        # Recall best path with min distance
        min_distance, _, best_path = heapq.heappop(self.results)
        figure = self.plot_distance_hisotry()
        return min_distance, best_path, figure

    def plot_distance_hisotry(self):
        
        x = np.arange(self.max_iter)
        y = np.array(self.path_distance_result)
        figure = plt.figure()
        plt.plot(x,y)
        plt.xlabel("Iterations")
        plt.ylabel("Distance")

                # Rendering the figure onto a canvas
        figure.set_size_inches(config.POSITION_LIMIT / figure.get_dpi(), (config.POSITION_LIMIT+50) / figure.get_dpi())
        figure.canvas.draw()

        # Get the RGB pixel values as a string and convert it to a numpy array
        width, height = figure.canvas.get_width_height()
        buffer = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        image = buffer.reshape((height, width, 3))

        return image  
    
    def initialize_adj_p_matrix(self,graph):
        
        dim = len(graph)
        distance_matrix = np.ones((dim,dim))*np.inf # N x N
        pheromone_matrix = np.ones((dim,dim)) * 0.2 # N x N

        for n,node in enumerate(graph):
            for nj, node_i in enumerate(graph):
                if nj > n:
                    distance_matrix[n,nj] = node.get_distance(node_i)
        
        return np.stack([distance_matrix,pheromone_matrix],-1) 
        
        
    def update_pheromone(self,ant_paths):
        n,m,_ = self.adj_pheromone_matrix.shape
        deltaTau = np.zeros((n,m))
        for n,(total_distance, _ ,ant_path) in enumerate(ant_paths):
            # update_loop.set_description(desc=f"update_loop {n}: ")
            result =  ant_path.result
            while result.child is not None:
                lid = self.graph_index[result]
                result = result.child
                childid = self.graph_index[result]
                if lid > childid:
                    deltaTau[childid,lid]  += self.Q / total_distance
                else:
                    deltaTau[lid,childid]  += self.Q / total_distance

        self.adj_pheromone_matrix[:,:,1] *= (1-self.p_decay) 
        self.adj_pheromone_matrix[:,:,1] += deltaTau

    def single_ant_path(self, ant, graph:list[Node]):
        path_result = PathResult()
        start_id = random.randint(0,len(graph)-1)
        curr_node = graph.pop(start_id)
        path_result.add_child(curr_node)
        while len(graph) > 1:
            
            prob_list = self.get_prob_list(curr_node,graph)
            curr_node = np.random.choice(graph, p=prob_list)
            graph.remove(curr_node)
            path_result.add_child(curr_node)
        path_result.add_child(graph[0])

        return path_result
    
    def get_prob_list(self,curr_node,allowed_graph):
        curr_index = self.graph_index[curr_node]

        propabiliy_list = np.zeros(len(allowed_graph))
        for n, node in enumerate(allowed_graph):
            node_id = self.graph_index[node]
            
            if curr_index > node_id:
                dist,pheromon = self.adj_pheromone_matrix[node_id,curr_index,:]
                curr_score = pheromon**self.alpha * (1/dist)**self.beta
                propabiliy_list[n] = curr_score
            else:
                dist,pheromon = self.adj_pheromone_matrix[curr_index,node_id,:]
                curr_score = pheromon**self.alpha * (1/dist)**self.beta
                propabiliy_list[n] = curr_score

        propabiliy_list = propabiliy_list / np.sum(propabiliy_list)
        return propabiliy_list

        
    def single_iteration(self,graph):
        self.path_result = []
        for ant in range(self.n_ants):
            result = self.single_ant_path(ant,deepcopy(graph))
            heapq.heappush(self.path_result, (result.total_distance,ant, result))
        self.update_pheromone(self.path_result)
        return heapq.heappop(self.path_result)

class ACSystem(BaseAntColonySystem):
    def __init__(self,):
        super().__init__()
