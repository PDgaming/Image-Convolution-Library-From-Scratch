import numpy as np

gaussian_kernel = np.array([[1, 2, 1],[2, 4,2],[1,2,1]]) / 16

edge_detection_kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

emboss_kernel = np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]])