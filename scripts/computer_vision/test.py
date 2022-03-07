import cv2
import numpy as np
import csv
import ast
import sys
from sift_template import cd_sift_ransac, cd_template_matching
from color_segmentation import cd_color_segmentation

# File paths
cone_csv_path = "./test_images_cone/test_images_cone.csv"
citgo_csv_path = "./test_images_citgo/test_citgo.csv"
localization_csv_path="./test_images_localization/test_localization.csv"

cone_template_path = './test_images_cone/cone_template.png'
citgo_template_path = './test_images_citgo/citgo_template.png'
localization_template_path='./test_images_localization/basement_fixed.png'

cone_score_path = './scores/test_scores_cone.csv'
citgo_score_path = './scores/test_scores_citgo.csv'
localization_score_path = './scores/test_scores_map.csv'


cone_test_path = "./test_images_cone/test10.jpg"
ctigo_test_path= "./test_images_citgo/citgo6.jpeg"
loco_test_path= "./test_images_localization/map_scrap4.png"

if __name__ == '__main__':
    cone = cv2.imread(cone_test_path)
    ctigo = cv2.imread(ctigo_test_path)
    loco = cv2.imread(loco_test_path)


    temp = cv2.imread(cone_template_path)
    temp_ctigo = cv2.imread(citgo_template_path)
    temp_loco  = cv2.imread(localization_template_path)
    # botttom_left, up_right = cd_sift_ransac(cone, temp)
    botttom_left, up_right = cd_sift_ransac(ctigo, temp_ctigo) # for sift
    # botttom_left, up_right = cd_sift_ransac(loco, temp_loco)
