import os
import uuid
import cv2 as cv

import pandas as pd

#file processed
# 202303031349 -->done
# 202303031347 --> done
# 202303031421 --> not process those are images collected in the robotics club room for training. 

old_csv = 'data/202303031421/labels.csv'
images = 'data/202303031421/images'
new_csv = "data/data_finale/labels1.csv"

def move_img(filename):
    unique_id = str(uuid.uuid4())
    unique_id = unique_id.split("-")
    unique_id = unique_id[4]
    new_filename = unique_id + ".jpg"
    img_path = f"{images}/{filename}"
    #print(img_path)
    image = cv.imread(img_path)
    cv.imwrite(f"data/data_finale/img1/{new_filename}", image)
    return new_filename


# reading the csv file
df = pd.read_csv(old_csv, header=None)

for filename in os.listdir(images):             
    if filename.endswith('.jpg'):
        id = move_img(filename)
        count = 0
        for count in range(0, len(df)):
            # updating the column value/data
            x=df.loc[count][0]
            # print(x)
            if x == filename:
                df = df.replace(x,id)
            count += count

# writing into the file
df.to_csv(new_csv, index=False,header=False)