import json
from os import path
from pathlib import Path
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats
import sys
from itertools import combinations 

with open('colors.csv') as colors_files:
    colors = colors_files.read().splitlines()

KEYS = [ 'dbs', 'servers', 'buses', 'langs', 'gates', 'monitors', 'discos', 'images']
NONSIZE_KEYS = ['images']
SIZE_KEYS = ["num_%s" % (k,) for k in KEYS if k not in NONSIZE_KEYS]
SIZE_KEYS.append("num_files")
SIZE_KEYS.append("num_dockers")
SIZE_KEYS.append("num_services")
SIZE_KEYS.append("num_ms")
SIZE_KEYS.append("size")
SIZE_KEYS.append("avg_size_service")
SIZES = {k: [] for k in SIZE_KEYS}

CLEANER = {}


DATA = {}

for key in KEYS:
    DATA[key] = [[], []]
    CLEANER[key] = [[''], {}]

CLEANER['langs'][0] += ['css', 'html', 'jupyternotebook']
CLEANER['langs'][1].update({'golang' : 'go'})

CLEANER['images'][0] += ['base']


def clean_data(data):
    for key in KEYS:
        for v in data[key]:
            data[key] = set(data[key]) - set(CLEANER[key][0])
            syn = CLEANER[key][1]
            data[key] = [syn[x] if x in syn else x for x in data[key]]
    data['num_ms'] = data['num_services']-data['num_dbs']-data['num_buses']-data['num_discos']-data['num_monitors']-data['num_gates']


num_services = num_ok = num_files = num_dockers = num_compose = 0


dockers = []
def analyze_data(data):
    global DATA, KEYS, NUMS, num_services, num_ok, num_files, num_dockers, num_compose
    clean_data(data)
    if not data['num_dockers'] or data['num_dockers'] > 250:
        return
    num_services += data['num_services']
    num_ok += data['num_dockers'] > 0
    num_files += data['num_files']
    num_dockers += data['num_dockers']
    num_compose += data['num_services'] > 0
    dockers.append(data['num_dockers'])
    for key in KEYS:
        DATA[key][0] += data[key]
        if data[key]:
            DATA[key][1].append(tuple(sorted(data[key])))
    for key in SIZE_KEYS:
        SIZES[key].append(data[key])

def analyze_all():
    repos = Path('results').glob('*.json')
    i = j = 0
    for source in repos:
        j += 1
        try:
            with open(str(source)) as json_file:
                data = json.load(json_file)
                analyze_data(data)
        except (UnicodeDecodeError, json.decoder.JSONDecodeError):
            i += 1
            pass
    
    print("TOTAL", j, "ERRORS", i)


def color_with_alpha(hex, alpha):
    hex = hex.lstrip('#')
    return [int(hex[i:i+2], 16)/256 for i in (0, 2, 4)] + [alpha]

def plot_scatter(name, *data, scale='linear', xlabel='', ylabel='', legend=[], colors=[]):
    alpha = 0.8
    for i,d in enumerate(data):
        if colors:
            plt.scatter(range(len(d)), d, s=500, marker='.', color=color_with_alpha(colors[i], alpha))
        else:
            plt.scatter(range(len(d)), d, s=500, marker='.')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(scale)
    plt.legend(legend)
    plt.savefig(f'plots/{name}.pdf', bbox_inches='tight')
    plt.cla()
    plt.clf()


def create_hist(name, b, *data, xlabel='', ylabel='', interval=False, legend=[], colors=[]):
    def autolabel(rects):
       for rect in rects:
        h = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.0*h, '%d'%int(h), ha='center', va='bottom')
    
    _, bins = np.histogram(data[0], bins=b)
    bin_counts = list(zip(bins, bins[1:]))
    x = []
    _, bin_end = bin_counts[0]
    x.append(f'≤{bin_end}')
    for bin_start, bin_end in bin_counts[1:-1]:
        if interval:
            x.append(f'{bin_start}-{bin_end}')
        else:
            x.append(f'{bin_end}')
    bin_start, _ = bin_counts[-1]
    x.append(f'>{bin_start}')
    x_pos = [i for i, _ in enumerate(x)]
    plt.xticks(x_pos, x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    width = 0.27
    l = len(data)
    for j, d in enumerate(data):
        y, _ = np.histogram(d, bins=b)
        x_offset = (j - l / 2) * width + width / 2
        x_pos = [i+x_offset for i, _ in enumerate(x)]
        if colors:
            rect = plt.bar(x_pos, y, width, color=colors[j])
        else:
            rect = plt.bar(x_pos, y, width)       
        #autolabel(rect)
      

    plt.legend(legend)
    plt.savefig(f'plots/{name}.pdf',bbox_inches='tight')
    plt.cla()
    plt.clf()

def plot_bar(name, *data, ticks=[], xlabel='', ylabel='', legend=[], colors=[]):
    l = len(data)
    width = 0.27
    for j, y in enumerate(data):
        x_offset = (j - l / 2) * width + width / 2
        x_pos = [i+x_offset for i, _ in enumerate(y)]
        if colors:
            rect = plt.bar(x_pos, y, width, color=colors[j])
        else:
            rect = plt.bar(x_pos, y, width)       
    plt.legend(legend)
    plt.xticks([i for i, _ in enumerate(ticks)], ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'plots/{name}.pdf',bbox_inches='tight')
    plt.cla()
    plt.clf()

def plot_barh(name, *data, ticks=[], xlabel='', ylabel='', legend=[], colors=[]):
    l = len(data)
    width = 0.50
    for j, y in enumerate(data):
        x_offset = (j - l / 2) * width + width / 2
        x_pos = [i+x_offset for i, _ in enumerate(y)]
        if colors:
            rect = plt.barh(x_pos, y, width, color=colors[j])
        else:
            rect = plt.barh(x_pos, y, width)       
    plt.legend(legend)
    plt.yticks([i for i, _ in enumerate(ticks)], ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'plots/{name}.pdf',bbox_inches='tight')
    plt.cla()
    plt.clf()

def plots():
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

    plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

    matplotlib.rc('font', **font)

    plot_scatter('size', [x/1000 for x in SIZES['size'] if x < 10**6], [x/1000 for x in SIZES['avg_size_service'] if x < 10**6], legend=['Project size', 'Microservice size'], xlabel='Repositories', ylabel='Size (MB)', colors=colors) 
    #create_hist(SIZES['size'], [0, 10**3, 10**4, 10**5, sys.maxsize])
    create_hist('num_services', [1, 3, 5, 7, 10, 15, 20, sys.maxsize], SIZES['num_dockers'], SIZES['num_services'], SIZES['num_ms'], interval=True, legend=['# Dockerfile', '# Compose services', '# Microservices'], ylabel='Occurences', colors=colors)
    #create_hist(, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, sys.maxsize])
    #create_hist(SIZES['num_services'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, sys.maxsize])
    #create_hist(SIZES['avg_size_service'], [0, 10**3, 5*10**3, 10**4, sys.maxsize])


analyze_all()
print(num_services, num_ok, num_files, num_dockers, num_compose)

'''
for key in KEYS:
    print(key)
    print(Counter(DATA[key][0]).most_common(50))
    print(Counter(DATA[key][1]).most_common(50))
    print('---')

'''
plots()

imagescomb = []
for data in DATA['images'][1]:
    imagescomb += [tuple(set(x)) for x in combinations(data, 2)]


data, labels = [], []
for x in Counter(imagescomb).most_common(20):
    data.append(x[1])
    labels.append('-'.join(x[0]))
plot_barh('images+comb', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=colors)

data, labels = [], []
for x in Counter(DATA['images'][0]).most_common(20):
    data.append(x[1])
    label = x[0]
    if '/' in label:
        parts = x[0].split('/')
        label = f'{parts[0]}/../{parts[-1]}'
    labels.append(label)
plot_barh('images', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=colors)

