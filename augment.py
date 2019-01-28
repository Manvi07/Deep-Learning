import os
import cv2
import Augmentor
import numpy as np

name = os.listdir("./Leaf/singleLeaf/")
input_images = []
output_images = []
print("loading_images")

p = Augmentor.Pipeline("./Leaf/singleLeaf/")
p.ground_truth("./Leaf/masks/")
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.sample(1000)
# input_images.append(img_X)
# cv2.imwrite("./Leaf/AugmentedLeaves/"+str(i), img_y)
