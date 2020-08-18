import json
from os import path
from pathlib import Path
from collections import Counter


KEYS = [ 'dbs', 'servers', 'buses', 'langs', 'gates', 'monitors', 'discos' ]
NUMS = ["num_%s" % (k,) for k in KEYS]
NUMS.append("num_files")
NUMS.append("num_dockers")
NUMS.append("num_services")

CLEANER = {}

DATA = {}

for key in KEYS:
    DATA[key] = [[], []]
    CLEANER[key] = [[''], {}]
    
CLEANER['langs'][0] += ['css', 'html', 'jupyternotebook']
CLEANER['langs'][1].update({'golang' : 'go'})

def clean_data(data):
    for key in KEYS:
        for v in data[key]:
            data[key] = set(data[key]) - set(CLEANER[key][0])
            syn = CLEANER[key][1]
            data[key] = [syn[x] if x in syn else x for x in data[key]]


num_services = num_ok = num_files = num_dockers = 0

def analyze_data(data):
    global DATA, KEYS, NUMS, num_services, num_ok, num_files, num_dockers
    clean_data(data)
    num_services += data['num_services']
    num_ok += data['num_services'] > 0
    num_files += data['num_files']
    num_dockers += data['num_dockers']
    for key in KEYS:
        DATA[key][0] += data[key]
        if data[key]:
            DATA[key][1].append(tuple(sorted(data[key])))

def analyze_all():
    repos = Path('results').glob('*.json')
    i = 0
    for source in repos:
        try:
            with open(str(source)) as json_file:
                data = json.load(json_file)
                analyze_data(data)
        except (UnicodeDecodeError, json.decoder.JSONDecodeError):
            print(source)
            i += 1
            pass
    
    print("ERRORS", i)


analyze_all()
print(num_services, num_ok, num_files, num_dockers)
for key in KEYS:
    print(key)
    print(Counter(DATA[key][0]).most_common(50))
    print(Counter(DATA[key][1]).most_common(50))
    print('---')