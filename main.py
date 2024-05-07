from utils import generate_node_map,plot_cv_map,draw_connection,generate_final_result
from ant_colony_system import BaseAntColonySystem
import config
import cv2
from datetime import datetime
if __name__ == "__main__":

    graph = generate_node_map(40)

    imagmap = plot_cv_map(graph,config.POSITION_LIMIT,config.POSITION_LIMIT)

    path_finder = BaseAntColonySystem(alpha=config.ALPHA,
                                      beta=config.BETA,
                                      max_iter=config.MAX_ITER,
                                      n_ants=config.N_ANTS,
                                      Q = config.Q,
                                      p_decay=config.P_DECAY,
                                      random_state=config.RANDOM_STATE)

    min_distance,best_path,figure = path_finder.calculate_path(graph)
    print(f"{min_distance=}")

    imagmap = draw_connection(best_path,imagmap,min_distance)
    final_image = generate_final_result(figure,imagmap)
    cv2.imshow("Final Result",final_image)
    cv2.waitKey(0)

    if config.SAVE_RESULT:
        filename = f"result_{datetime.today().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename,(final_image*255).astype(int))
