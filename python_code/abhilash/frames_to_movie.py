# Have to deactivate the conda env and activate the my_env to work with cv2

import cv2
import os

# Folder containing frames
folder = "frames"

# Output video file
output = "output.mp4"

# Frames per second
fps = 30

# Get sorted list of image files
images = sorted([img for img in os.listdir(folder) if img.endswith(".png")])

# Read first image to get size
first_frame = cv2.imread(os.path.join(folder, images[0]))
height, width, _ = first_frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output, fourcc, fps, (width, height))

# Write frames
for image in images:
    frame = cv2.imread(os.path.join(folder, image))
    video.write(frame)

video.release()

print("Video saved as", output)
