import git
from os import path
from pathlib import Path
import os
import dockerfile
from collections import Counter
import nltk
import sys
import string
import subprocess
import json
import shutil
import yaml
import Levenshtein

with open('db.csv') as db_file:
    dbs = [db.lower() for db in db_file.read().splitlines()]
with open('bus.csv') as bus_file:
    buses = [bus.lower() for bus in bus_file.read().splitlines()]
with open('lang.csv') as lang_file:
    langs = [lang.lower() for lang in lang_file.read().splitlines()]
with open('server.csv') as server_file:
    servers = [server.lower() for server in server_file.read().splitlines()]

def are_similar(name, candidate):
    return name == candidate or candidate in name
   
def match_one(name, l):
    for candidate in l:
        if are_similar(name, candidate):
            return [candidate]
    return []

def match_alls(names, l):
    alls = set()
    for name in names:
        alls.update(match_one(name, l))
    return list(alls)

def match_ones(names, l):
    for name in names:
        res = match_one(name, l)
        if res:
            return res
    return []

  
def clone(repo_url, full_repo_name):
    username, repo_name = full_repo_name.split('/')
    workdir = path.join("temp", username)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    full_workdir = path.join(workdir, repo_name)
    if not path.exists(full_workdir):
        print('-cloning repo')
        endpoint = 'https://api.github.com/repos/%s' % (full_repo_name,)
        p1 = subprocess.run(['curl', endpoint], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,)
        size = json.loads(p1.stdout.decode("utf-8"))['size']
        print('repo size', '%dM' % (size/1000,))
        if size < 50000:
            git.Git(workdir).clone(repo_url)
        else:
            print('repo too big')
            return None
    else:
        print('repo already cloned')
    return full_workdir

def locate_files(workdir, filename):
    print('-locating ', filename)
    res = []
    for df in Path(workdir).rglob(filename):
        df = str(df)
        res.append(df.split(workdir)[-1])
    return res

def get_words(data, unique=False):
    data = data.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    data = data.translate(str.maketrans(string.digits, ' '*len(string.digits)))
    data = data.lower()
    words = [w for w in nltk.word_tokenize(data) if len(w) > 2]
    if unique:
        words = set(words)
    return words

def keywords(data, n=5):
    words = get_words(data)
    counter = Counter(words)
    most_commons = [x[0] for x in counter.most_common(n)]
    return most_commons

def analyze_languages(workdir):
    print('-analyzing languages')
    result = subprocess.run(['github-linguist', workdir], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    return [r.split('  ')[1].lower().replace(' ', '') for r in output.split('\n')[:-1] if float(r.split('%')[0]) > 10]

def analyze_dockerfile(workdir, df):
    print('-analyzing dockerfile', df)
    commands = dockerfile.parse_file(workdir+df)
    runs = ''
    analysis = {'path': df, 'cmd': '', 'cmd_keywords': []}
    for command in commands:
        if command.cmd == 'from':
            analysis['from'] = command.value[0].split(':')[0]
            analysis['from_full'] = command.value[0]
        if command.cmd == 'run':
            runs += '%s ' % (' '.join(command.value),)
        if command.cmd == 'cmd':
            analysis['cmd'] = ' '.join(command.value)
            analysis['cmd_keywords'] = keywords(analysis['cmd'])
        analysis['keywords'] = keywords(runs)
    if 'from' not in analysis:
        return analysis
    analysis['dbs'] = match_one(analysis['from'], dbs) \
                    or match_ones(get_words(analysis['cmd']), dbs) \
                    or match_ones(get_words(runs), dbs)
    analysis['buses'] = match_one(analysis['from'], buses) \
                    or match_ones(get_words(analysis['cmd']), buses) \
                    or match_ones(get_words(runs), buses)
    analysis['servers'] = match_one(analysis['from'], servers) \
                    or match_ones(get_words(analysis['cmd']), servers) \
                    or match_ones(get_words(runs), servers)
    analysis['langs'] = match_one(analysis['from'], langs) \
                    or match_ones(get_words(analysis['cmd']), langs) \
                    or match_ones(get_words(runs), langs)
    return analysis

def analyze_file(workdir, f):
    print('-analyzing file', f)
    analysis = {'path': f}
    with open(workdir+f) as fl:
        data = ' '.join(fl.read().splitlines())
        analysis['dbs'] = match_alls(get_words(data), dbs)
        analysis['buses'] = match_alls(get_words(data), buses)
        analysis['servers'] = match_alls(get_words(data), servers)
    return analysis

def check_shared_db(analysis):
    db_services = set(analysis['detected_dbs']['services'])
    dependencies = []
    for service in analysis['services']:
        dependencies += set(service['depends_on']) & db_services
    return len(set(dependencies)) != len(dependencies)

def analyze_docker_compose(workdir, dc):
    print('-analyzing docker-compose')
    with open(workdir+dc) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        analysis = {'path': dc, 'num_services': 0, 'services': [], 'detected_dbs': { 'num' : 0, 'names': [], 'services': [], 'shared_dbs' : False} }
        services = []
        detected_dbs = []
        if 'services' not in data:
            return analysis
        for name, service in data['services'].items():
            s = {}
            s['name'] = name
            if 'image' in service:
                s['image'] =  service['image'].split(':')[0]
                s['image_full'] =  service['image']
            else:
                s['image'] = service['build']
                s['image_full'] =  service['build']

            s['dbs'] = match_one(s['image'], dbs)
            if s['dbs']:
                detected_dbs.append({'service' : name, 'name': s['dbs'][0]})
            s['buses'] = match_one(s['image'], buses)
            s['servers'] = match_one(s['image'], servers)

            if 'depends_on' in service:
                if isinstance(service['depends_on'], dict):
                    s['depends_on'] = list(service['depends_on'].keys())
                else:
                    s['depends_on'] = service['depends_on']
            elif 'links' in service:
                s['depends_on'] = list(service['links'])
            else:
                s['depends_on'] = []
            services.append(s)
        analysis['services'] = services
        analysis['num_services'] = len(services)
        analysis['detected_dbs'] = {'num': len(detected_dbs), \
                                     'names' : list({db['name'] for db in detected_dbs}), \
                                     'services' : [db['service'] for db in detected_dbs]}
        analysis['detected_dbs']['shared_dbs'] = check_shared_db(analysis)

    return analysis

def synthetize_data(analysis):
    keys = ['dbs', 'langs', 'servers', 'buses']
    def add_data(data):
        for d in data:
            for k in keys:
                if k in d:
                    analysis[k].update(d[k])

    for k in keys:
        analysis[k] = set()
    
    add_data(analysis['files'])
    add_data(analysis['structure']['services'])
    add_data(analysis['dockers'])
    analysis['num_services'] = analysis['structure']['num_services']
    analysis['shared_dbs'] = analysis['structure']['detected_dbs']['shared_dbs']
    analysis['langs'].update(analysis['languages'])
    analysis['num_dockers'] = len(analysis['dockers'])

    for k in keys:
        analysis['num_%s' % (k,)] = len(analysis[k])
        analysis[k] = list(analysis[k])


def analyze_repo(url):
    analysis = {'url' : url}
    analysis['name'] = url.split('.git')[0].split('git://github.com/')[-1]
    print('analyzing', analysis['name'])
    outfile = path.join('results', analysis['name'].replace('/', '#'))
    outfile = "%s.json" % (outfile,)
    if True:#not path.exists(outfile):
        workdir = clone(url, analysis['name'])
        if not workdir:
            return 
        analysis['languages'] = analyze_languages(workdir)
        dfs = locate_files(workdir, 'Dockerfile')
        dockers_analysis = []
        for df in dfs:
            dockers_analysis.append(analyze_dockerfile(workdir, df))
        analysis['dockers'] = dockers_analysis
        dc = locate_files(workdir, 'docker-compose.yml')
        analysis['structure'] = {'path': dc, 'num_services': 0, 'services': [], 'detected_dbs': { 'num' : 0, 'names': [], 'services': [], 'shared_dbs' : False} }

        if len(dc):
            dc = dc[0]
            analysis['structure'] = analyze_docker_compose(workdir, dc)
        fs = locate_files(workdir, 'requirements.txt')
        fs += locate_files(workdir, '*.gradle')
        fs += locate_files(workdir, 'pom.xml')
        file_analysis = []
        for f in fs:
            file_analysis.append(analyze_file(workdir, f))
        analysis['files'] =  file_analysis
        synthetize_data(analysis)
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=4)
        #shutil.rmtree(path.dirname(workdir))
    else:
        print('skipped')

def analyze_all():
    with open('repos.csv', 'r') as f:
       for repo in f:
            analyze_repo(repo)


analyze_all()