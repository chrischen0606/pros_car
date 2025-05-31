import cv2
import os
import yaml
import numpy as np

# Read PGM file (as grayscale)
image = cv2.imread('map01.pgm', cv2.IMREAD_UNCHANGED)

# Display image
# cv2.imshow('PGM Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
current_dir = os.path.dirname(os.path.abspath(__file__))
# map_path = os.path.join(current_dir, 'map01')
yaml_path = os.path.join(current_dir, 'map01.yaml')
with open(yaml_path, 'r') as f:
    map_metadata = yaml.safe_load(f)
    
R_align = np.array([
    [1,  0,  0],   # x_map → x_cam
    [0,  0, -1],   # y_map → z_cam
    [0,  1,  0]    # z_map → -y_cam
])
T_align = np.eye(4)
T_align[0:3, 0:3] = np.linalg.inv(R_align)
print(T_align)