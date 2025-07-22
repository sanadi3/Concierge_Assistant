import csv
import random

rows = 146
min_value = 1.00
max_value = 100.00

input_file = 'data/PoI.csv'
output_file = 'PoIRandom.csv'


with open(input_file, mode='r', newline='') as infile, \
    open(output_file, mode='w', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        row[12] = random.uniform(min_value, max_value)
        row[13] = random.uniform(min_value, max_value)

        writer.writerow(row)

