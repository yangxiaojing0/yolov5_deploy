# # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""原版"""

# import pprint

# import requests

# DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
# IMAGE = "zidane.jpg"

# # Read image
# with open(IMAGE, "rb") as f:
#     image_data = f.read()

# response = requests.post(DETECTION_URL, files={"image": image_data}).json()

# pprint.pprint(response)

"""Perform test request"""
import pprint

import requests

DETECTION_URL = "http://localhost:8801/v1/object-detection/yolov5s"
TEST_IMAGE = "/workspace/yolov5_deploy/data/images/bus.jpg"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)

