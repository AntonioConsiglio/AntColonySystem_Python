from utils import (generate_node_map,plot_cv_map,
                   draw_connection,generate_final_result,
                   draw_pheromone, save_experiment_result)
from ant_colony_system import BaseAntColonySystem,ElitarianACS
import config
import cv2

if __name__ == "__main__":

    graph = generate_node_map(40)

    imagmap = plot_cv_map(graph,config.POSITION_LIMIT,config.POSITION_LIMIT)

    # path_finder = BaseAntColonySystem(alpha=config.ALPHA,
    #                                   beta=config.BETA,
    #                                   max_iter=config.MAX_ITER,
    #                                   n_ants=config.N_ANTS,
    #                                   Q = config.Q,
    #                                   p_decay=config.P_DECAY,
    #                                   random_state=config.RANDOM_STATE)
    
    path_finder = ElitarianACS(alpha=config.ALPHA,
                                      beta=config.BETA,
                                      max_iter=config.MAX_ITER,
                                      n_ants=config.N_ANTS,
                                      Q = config.Q,
                                      p_decay=config.P_DECAY,
                                      random_state=config.RANDOM_STATE)
    

    for iter,min_distance,best_path,figure, best_iteration in  path_finder.calculate_path(graph):
        print(f"Best iteration: {best_iteration} -> {min_distance=:.1f}")
        imgmap = draw_pheromone(graph,imagmap,path_finder.adj_pheromone_matrix[...,1])
        imgmap = draw_connection(best_path,imgmap,min_distance,best_iteration=best_iteration)
        final_image = generate_final_result(figure,imgmap)
        

        if config.SHOW_RESULT:
            cv2.imshow("Final Result",final_image)
            cv2.waitKey(10)

        if config.SAVE_RESULT:
            save_experiment_result(final_image,iter)