# ACS implemented in python

![example](https://github.com/AntonioConsiglio/AntColonySystem_Python/blob/main/gif/example.gif)

ACS (Ant Colony System) is a metaheuristic algorithm inspired by the foraging behavior of ants. It's designed to find high-quality solutions to the Traveling Salesman Problem, a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the original city.



The algorithm has 4 steps:



1) Initialization:

    Initially, a colony of artificial ants is placed randomly on different cities. Parameters like the number of ants, pheromone evaporation rate,pheromone trail initialization, and heuristic information are set.



2) Ant Movement:

    Each ant starts constructing a solution by iteratively selecting the next city to visit. The probability of selecting a city depends on the amount of pheromone on the edge and the desirability of the city based on heuristic information (e.g., distance between cities). Ants follow a probabilistic rule known as the state transition rule to decide their next move.



3) Pheromone Update:

    After all ants complete their tours, pheromone update occurs. The amount of pheromone deposited on each edge is proportional to the quality of the solutions found. Shorter tours result in more pheromone deposition. Pheromone evaporation is also applied to prevent stagnation and to give a chance to explore new paths.



4) Iteration:

    Steps 2 and 3 are repeated iteratively until a termination condition is met (e.g., a maximum number of iterations or a satisfactory solution is found).
