import cv2
import os
import pandas as pd

image_folder = '.'
video_name = '2000007889_080219_PR_U_made_from_images_19jul.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
file_number = [int(w.replace("_infer.png", '')) for w in images]

d = {'sn': file_number, 'file_name' : images }
df= pd.DataFrame(d).sort_values(by=['sn'])
#print(df['file_name'])

#print(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))

for image in df['file_name']:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
