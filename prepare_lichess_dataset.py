import csv
import sys
import json
import time

from src.core import format_time
from src.loading.raw_data_gather import gather
from src.patches import LICHESS_PATH, LICHESS_DATASET_PATH

if len(sys.argv) < 2:
    print("Usage: python prepare_lichess_dataset.py <size>")
    exit(0)

SIZE = int(sys.argv[1])

gather("https://database.lichess.org/",
       "lichess_db_eval.jsonl.zst",
       LICHESS_PATH)

with open(LICHESS_PATH, "r") as infile:
    with open(LICHESS_DATASET_PATH, "w", newline='') as outfile:
        writer = csv.writer(outfile)
        counter = 0
        t0 = time.time()
        while counter < SIZE:
            node = json.loads(infile.readline())
            fen = node["fen"]
            result = 0
            depth = 0
            for pv in node["evals"]:
                if pv["depth"] > depth:
                    depth = pv["depth"]
                    if 'cp' in pv['pvs'][0]:
                        result = pv['pvs'][0]['cp']
                    if 'mate' in pv['pvs'][0]:
                        result = pv['pvs'][0]['mate']
                        result = "M" + str(result)
            writer.writerow([fen,0,0,0, result])
            counter+=1
            if counter % 100000 == 0:
                deltat = time.time() - t0
                eta = deltat * (SIZE - counter) / counter
                print(str(counter) + "/" + str(SIZE) + " t:" + format_time(deltat) + " eta: " + format_time(eta))
