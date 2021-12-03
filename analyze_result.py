import json
from os import path
from pathlib import Path
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import sys
import pickle
import csv
import argparse
from tabulate import tabulate
import statistics as stats

from itertools import combinations, product

with open('./consts/colors.csv') as colors_files:
    COLORS = colors_files.read().splitlines()

KEYS = [ 'dbs', 'servers', 'buses', 'langs', 'gates', 'monitors', 'discos', 'images']
NONSIZE_KEYS = ['images']
SIZE_KEYS = ["num_%s" % (k,) for k in KEYS if k not in NONSIZE_KEYS]
SIZE_KEYS.append("num_files")
SIZE_KEYS.append("num_dockers")
SIZE_KEYS.append("num_services")
SIZE_KEYS.append("num_ms")
SIZE_KEYS.append("size")
SIZE_KEYS.append("avg_size_service")
SIZE_KEYS.append("commiters")
SIZE_KEYS.append("shared_dbs")

SIZES = {k: [] for k in SIZE_KEYS}

CLEANER = {}


DATA = {}
DEP_GRAPHS = []

for key in KEYS:
    DATA[key] = [[], [], []]
    CLEANER[key] = [[''], {}]

CLEANER['langs'][0] += ['css', 'html', 'jupyternotebook', 'vue', 'dockerfile', 'scratch', 'bash', 'shell', 'makefile']
CLEANER['langs'][1].update({'golang' : 'go', 'gcc' : 'c', 'cmake': 'c'})

CLEANER['images'][0] += ['base']
CLEANER['servers'][0] += ['mongoose', 'zookeeper']
CLEANER['dbs'][0] += ['max', 'zookeeper', 'db']
CLEANER['dbs'][1].update({'sql' : 'mysql'})

CLEANER['monitors'][0] += ['monitoring']
CLEANER['gates'][0] += ['gateway', 'loadbalancer', 'loadbalancing']

#CLEANER['gates'][1].update({'gateway' : '(G) nginx', 'loadbalancer': '(G) zuul', 'loadbalancing': ' (G) kong'}
CLEANER['gates'][1].update({'nginx' : ' nginx (G)', 'zuul': 'zuul (G)', 'kong' : 'kong (G)', 'linkerd' : 'linkerd (G)'})

parser = argparse.ArgumentParser()
parser.add_argument("-f", dest='filter_file', type=str, help="Filter file", required=True)
parser.add_argument("-s", dest='min_services', type=int, help="Min services", default=0)
args = parser.parse_args()

def clean_data(data):
    for key in KEYS:
        data[key] = set(data[key]) - set(CLEANER[key][0])
        syn = CLEANER[key][1]
        data[key] = [syn[x] if x in syn else x for x in data[key]]

    data['num_ms'] = max(2, data['num_services']-data['num_dbs']-data['num_buses']-data['num_discos']-data['num_monitors']-data['num_gates'])

def analyze_data(data):
    global DATA, KEYS
    clean_data(data)
    if data['num_dockers'] > 30:
        return False
    # discard repos with microservices < args.min_services
    num_ms = data['num_ms']
    if num_ms < args.min_services:
         return False
    for key in KEYS:
        DATA[key][0] += data[key]
        if data[key]:
            DATA[key][1].append(tuple(sorted(data[key])))
        DATA[key][2].append(tuple(sorted(data[key])))

    for key in SIZE_KEYS:
        SIZES[key].append(data[key])

    # save the dependencies graphs
    if data['structure']:
        DEP_GRAPHS.append({'full': data['structure']['dep_graph_full'],
                           'micro': data['structure']['dep_graph_micro']})
    return True

def analyze_all():
    global DATA, SIZES
    if path.exists('temp/SIZES') and path.exists('temp/DATA'):
        print('retriving from disk')
        with open('temp/SIZES', 'rb') as f:
            SIZES = pickle.load(f)
        with open('temp/DATA', 'rb') as f:
            DATA = pickle.load(f)
        return
    
    include = set()
    
    with open(args.filter_file, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            while True:
                try:
                    line = next(reader)
                    #print("Line",line)
                    if line[2] == 'D' or line[2] == 'M':
                        include.add(line[1])
                except UnicodeDecodeError as e:
                    print(e)
                except StopIteration:
                    break

    print("INCLUDING", len(include), " REPOS")
    
    repos = Path('results').glob('*.json')
    i = j = l = 0
    for source in repos:
        j += 1
        try:
            with open(str(source)) as json_file:
                data = json.load(json_file)
                if data['url'] and data['url'] in include:
                    if analyze_data(data):
                        l += 1
                
        except (UnicodeDecodeError, json.decoder.JSONDecodeError):
            i += 1
            pass

    with open('temp/SIZES', 'wb') as f:
        pickle.dump(SIZES, f)
    with open('temp/DATA', 'wb') as f:
        pickle.dump(DATA, f)
    print('writed on disk')
    print("TOTAL", j, "ANALYZED", l, "ERRORS", i)
    print("DATA len", len(DATA))
    print("DEP_GRAPHS len", len(DEP_GRAPHS))

def color_with_alpha(hex, alpha):
    hex = hex.lstrip('#')
    return [int(hex[i:i+2], 16)/256 for i in (0, 2, 4)] + [alpha]

def plot_scatter(name, *data, x=[], scale='linear', xlabel='', ylabel='', legend=[], colors=[]):
    alpha = 0.8
    if not x:
        x = range(len(d))
    for i,d in enumerate(data):
        if colors:
            plt.scatter(x, d, s=500, marker='.', color=color_with_alpha(colors[i], alpha))
        else:
            plt.scatter(x, range(len(d)), d, s=500, marker='.')

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
            x.append(f'{bin_start+1}-{bin_end}')
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
    width = 0.27
    l = len(data)
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
    width = 0.50
    l = len(data)
    for d in data:
        for j, y in enumerate(data):
            y = list(reversed(y))
            x_offset = (j - l / 2) * width + width / 2
            x_pos = [i+x_offset for i, _ in enumerate(y)]
            if colors:
                rect = plt.barh(x_pos, y, width, color=colors[j])
            else:
                rect = plt.barh(x_pos, y, width)       
    plt.legend(legend)
    plt.yticks([i for i, _ in enumerate(ticks)], list(reversed(ticks)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'plots/{name}.pdf',bbox_inches='tight')
    plt.cla()
    plt.clf()

def plot_pie(name, b, data, colors=[], labels=None, interval=True):
    def label(pct, data):
        val = int(pct/100*sum(data))
        return "{:.1f}%".format(pct)

    data, bins = np.histogram(data, bins=b)
    bin_counts = list(zip(bins, bins[1:]))
    x = []
    _, bin_end = bin_counts[0]
    x.append(f'≤{bin_end}')
    for bin_start, bin_end in bin_counts[1:-1]:
        if interval:
            x.append(f'{bin_start+1}-{bin_end}')
        else:
            x.append(f'{bin_end}')
    bin_start, _ = bin_counts[-1]
    x.append(f'>{bin_start}')
    if colors:
        rect = plt.pie(data, labels=x if not labels else labels, colors=colors, autopct=lambda pct: label(pct, data))
    else:
        rect = plt.pie(data, labels=x if not labels else labels, autopct=lambda pct: label(pct, data))

    plt.savefig(f'plots/{name}.pdf',bbox_inches='tight')
    plt.cla()
    plt.clf()

plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

def nice_colors(*indexes):
    return [color for i, color in enumerate(COLORS) if i in indexes]

def plots():
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 26}

    matplotlib.rc('font', **font)


    create_hist('size', [1, 7, 15, 25, 50, sys.maxsize], [x/1000 for x in SIZES['size']], [x/1000 for x in SIZES['avg_size_service']], interval=True, legend=['Project size', ''
                                                                                                                                                                              'Service size'], ylabel='Occurrences', xlabel='Size (MB)', colors=nice_colors(0, 5))
    create_hist('num_services', [1, 4, 6, 8, 10, 15, 20, sys.maxsize], SIZES['num_dockers'], SIZES['num_services'], SIZES['num_ms'], interval=True, legend=['# Dockerfile', '# Compose services', '# Microservices'], ylabel='Occurrences', colors=nice_colors(0, 3, 5))
    
    #plot_scatter('size-services', [x/1000 for x in SIZES['size']], x=[x/1000 for x in SIZES['avg_size_service']], ylabel='Project Size (MB)', xlabel='Avg Service Size (MB)', colors=nice_colors(0, 5)) 

    imagescomb = []
    for data in DATA['images'][1]: 
        imagescomb += [tuple(sorted(x)) for x in combinations(set(data), 2)]
        imagescomb += [tuple(sorted(x)) for x in combinations(set(data), 3)]
    print(DATA['images'][1])
    data, labels = [], []
    for x in Counter(imagescomb).most_common(20):
        data.append(x[1])
        images = []
        for i in x[0]:
            images.append(i.split('/')[-1])
        labels.append('+'.join(images))

    plot_barh('images-comb', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)

    data, labels = [], []
    for x in Counter(DATA['images'][0]).most_common(20):
        data.append(x[1])
        label = x[0]
        if '/' in label:
            parts = x[0].split('/')
            #label = f'{parts[0]}/../{parts[-1]}'
            label = parts[-1]
        labels.append(label)
    plot_barh('images', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)
    print(labels, data)
    data, labels = [], []

    for x in Counter(DATA['langs'][0]).most_common(20):
        data.append(x[1])
        label = x[0]
        labels.append(label)

    plot_barh('langs', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)

    comb = []
    for data in DATA['langs'][1]:
        comb += [tuple(sorted(x)) for x in combinations(set(data), 2)]
        comb += [tuple(sorted(x)) for x in combinations(set(data), 3)]


    data, labels = [], []
    for x in Counter(comb).most_common(10):
        data.append(x[1])
        labels.append('+'.join(x[0]))
    plot_barh('langs-comb', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)
    plot_pie('committers', [1, 3, 5, 8, sys.maxsize], [x/max(1,SIZES['num_ms'][i]) for i,x in enumerate(SIZES['commiters'])], labels=['Micro (≤3)', 'Small (4-5)', 'Medium (6-8)', 'Large (>8)'],  interval=True, colors=COLORS[1:])
    create_hist('num_langs', [1, 2, 3, 4, 5,  sys.maxsize], [x/max(1,SIZES['num_langs'][i]) for i, x in enumerate(SIZES['num_ms'])], legend=['# Languages', '# Microservices'], interval=False , ylabel='Occurences', colors=nice_colors(0, 3, 5))
    plot_barh('langs-comb', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)

    data, labels = [], []
    for x in Counter(DATA['buses'][0]).most_common(20):
        data.append(x[1])
        label = x[0]
        labels.append(label)

    plot_barh('buses', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)

    data, labels = [], []
    for x in Counter(DATA['discos'][0]+DATA['gates'][0]).most_common(20):
        data.append(x[1])
        label = x[0]
        labels.append(label)

    plot_barh('disco+gates', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)

    data, labels = [], []
    for x in Counter(DATA['monitors'][0]).most_common(20):
        data.append(x[1])
        label = x[0]
        labels.append(label)

    plot_barh('monitors', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)

    data, labels = [], []
    for x in Counter(SIZES['shared_dbs']).most_common(2):
        data.append(x[1])
        label = x[0]
        labels.append(label)
    plot_barh('shared_dbs', data, ticks=labels, xlabel='Occurrences', ylabel='', legend=[], colors=COLORS)

def tables():

    servers = DATA['servers'][2]
    dbs = DATA['dbs'][2]

    comb = []
    for i, s in enumerate(servers):
        comb  += [tuple(set(x)) for x in product(s, dbs[i])]

    mostservers = [x for x,_ in Counter(DATA['servers'][0]).most_common(10)]
    mostdb = [x for x,_ in Counter(DATA['dbs'][0]).most_common(10)]
    print(Counter(DATA['servers'][0]).most_common(10))
    print(Counter(DATA['dbs'][0]).most_common(10))

    comb = []

    for i, s in enumerate(servers):
        s = set(s) & set(mostservers)
        d = set(dbs[i]) & set(mostdb)
        comb  += [tuple(x) for x in product(s, d)]

    data = Counter(comb).most_common(100)

    print(data)

    def search_tuple(t, data):
        for t1, v in data:
            if t1 == t:
                return v
        return 0

    for d in mostdb:
        print(f'& \\textit{{{d}}}', end=' ')

    print('\\\\ \\hline')
    totdb = [0] * 10
    for s in mostservers:
        print(f'\\textit{{{s}}}', end=' ')
        tots = 0
        for i, d in enumerate(mostdb):
            v = search_tuple((s,d), data)
            tots += v
            totdb[i] += v
            print(f'& ${search_tuple((s,d), data)}$', end=' ')
        print(f' & ${tots}$\\\\ \\hline')

    for v in totdb:
        print(f' & ${v}$', end=' ')

    print(f'\\\\ \\hline')


def dep_graphs_tables():
    print("Dependencies graphs table")
    values = {'full': {'nodes': [], 'edges': [], 'avg_deps_per_service': [],
              'acyclic': [], 'longest_path': []},
              'micro': {'nodes': [], 'edges': [], 'avg_deps_per_service': [],
              'acyclic': [], 'longest_path': []}}
    table = []
    for dg_type in ["full", "micro"]:
        for dg in DEP_GRAPHS:
            dep_graph = dg[dg_type]
            values[dg_type]['nodes'].append(dep_graph['nodes'])
            values[dg_type]['edges'].append(dep_graph['edges'])
            values[dg_type]['avg_deps_per_service'].append(dep_graph['avg_deps_per_service'])
            values[dg_type]['acyclic'].append(dep_graph['acyclic'])
            values[dg_type]['longest_path'].append(dep_graph['longest_path'])
        table.append([dg_type,
                      stats.mean(values[dg_type]['nodes']),
                      stats.mean(values[dg_type]['edges']),
                      stats.mean(values[dg_type]['avg_deps_per_service']),
                      len([ac for ac in values[dg_type]['acyclic'] if ac == False]),
                      stats.mean(values[dg_type]['longest_path'])])
    headers = ['Avg nodes', 'Avg edges', 'Avg avg deps per service', 'Num cyclic', 'Avg longest path']
    print(tabulate(table, headers=headers, floatfmt=".2f", tablefmt="latex"))



analyze_all()
print('all:', len(SIZES['num_services']), \
    '#services:', sum(SIZES['num_services']), \
    '#ok:', len([None for x in SIZES['num_services'] if x > 2]), \
    '#files:', sum(SIZES['num_files']), \
    '#dockers:', sum(SIZES['num_dockers']), \
    '#compose:', len([None for x in SIZES['num_services'] if x > 0]))
'''
for key in KEYS:
    print(key)
    print(Counter(DATA[key][0]).most_common(50))
    print(Counter(DATA[key][1]).most_common(50))
    print('---')

'''

plots()
tables()

dep_graphs_tables()

print("\n***STATS***\n")
for k, v in SIZES.items():
    if k == 'shared_dbs':
        break
    print(k.upper())
    a = np.array(v)
    print('mean', np.mean(a), 'std', np.std(a), 'min', np.min(a), 'max', np.max(a), np.percentile(a, 75))


a =  np.array([x/max(1,SIZES['num_langs'][i]) for i, x in enumerate(SIZES['num_ms'])])
print('mean', np.mean(a), 'std', np.std(a), 'min', np.min(a), 'max', np.max(a), np.percentile(a, 30))

print(Counter(DATA['discos'][0]).most_common(20))

print(Counter(DATA['monitors'][1]).most_common(20))


print(len([x for x in SIZES['shared_dbs'] if x]))

print(max([x/max(1,SIZES['num_ms'][i]) for i,x in enumerate(SIZES['commiters'])]))