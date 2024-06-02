from scenedetect import detect, AdaptiveDetector
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import os

video_local_path = "./keyframes_test/test_royal.mp4"
content_list = detect(video_local_path, AdaptiveDetector())


frame_num = (content_list[0][1] - content_list[0][0]).frame_num
first_scene_frame_index = frame_num // 2 + content_list[0][0].frame_num


video = cv2.VideoCapture(video_local_path)


video.set(cv2.CAP_PROP_POS_FRAMES, first_scene_frame_index)
_, first_scene_frame = video.read()

