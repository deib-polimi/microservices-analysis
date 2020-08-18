import json
from os import path
from pathlib import Path
from collections import Counter

num_services = 0
num_ok = 0
num_files = 0
num_dockers = 0
dbs_flat = []
dbs = []
langs = []
langs_flat = []

def analyze_data(data):
    global num_services, num_ok, num_files, num_dockers, dbs, dbs_flat, langs, langs_flat
    num_services += data['num_services']
    num_ok += 1 if data['num_services'] else 0
    num_files += data['num_files']
    num_dockers += data['num_dockers']
    dbs += data['dbs']
    dbs_flat.append(tuple(sorted(data['dbs'])))
    data['langs'] = list({'go' if x == 'golang' else x for x in data['langs']})
    langs += data['langs']
    langs_flat.append(tuple(sorted(data['langs'])))

def analyze_all():
    repos = Path('results').glob('*.json')
    for source in repos:
        with open(source) as json_file:
            data = json.load(json_file)
            analyze_data(data)


analyze_all()
print(num_services, num_ok, num_files, num_dockers)
cdbs = Counter(dbs)
cdbs_flat = Counter(dbs_flat)
clangs = Counter(langs)
clangs_flat = Counter(langs_flat)
print(cdbs.most_common(10))
print(cdbs_flat.most_common(10))
print(clangs.most_common(10))
print(clangs_flat.most_common(10))