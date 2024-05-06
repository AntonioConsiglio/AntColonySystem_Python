from utils import generate_node_map,plot_cv_map,draw_connection
from ant_colony_system import BaseAntColonySystem
import cv2

if __name__ == "__main__":
    graph = generate_node_map(30)

    imagmap = plot_cv_map(graph,400,400)

    path_finder = BaseAntColonySystem(alpha=1,
                                      beta=1,
                                      max_iter=100,
                                      n_ants=20,
                                      Q = 1000,
                                      p_decay=0.02,
                                      random_state=42)

    min_distance,best_path = path_finder.calculate_path(graph)
    print(f"{min_distance=}")

    imagmap = draw_connection(best_path,imagmap,1)
    cv2.imshow("Image Map",imagmap)
    cv2.waitKey(0)