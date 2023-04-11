import os
import csv
filenames = os.listdir("./data/2023_04_10_12_42/images")

with open ('./names.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for file_name in filenames:
        csv_writer.writerow((file_name))