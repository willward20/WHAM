import os
import uuid
import cv2 as cv

import pandas as pd
train_and_deploy/data/202303031349
old_csv = '../data/202303031349/labels.csv'
images = '../data/202303031349//images'
new_csv = "../data/data_finale/labels.csv"


def move_img(filename):
    unique_id = str(uuid.uuid4())
    unique_id = unique_id.split("-")
    unique_id = unique_id[4]
    new_filename = unique_id + ".jpg"
    img_path = f"{images}/{filename}"
    # print(img_path)
    image = cv.imread(img_path)
    cv.imwrite(f"data_finale/img/{new_filename}", image)
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