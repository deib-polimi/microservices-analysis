import git
from os import path
from pathlib import Path
import os
import dockerfile
from collections import Counter
import nltk
nltk.download('punkt')
import string
import subprocess
import json
import shutil
import yaml
import csv
from filelock import Timeout, FileLock
import networkx as nx


with open('./consts/db.csv') as db_file:
    dbs = [db.lower() for db in db_file.read().splitlines()]
with open('./consts/bus.csv') as bus_file:
    buses = [bus.lower() for bus in bus_file.read().splitlines()]
with open('./consts/lang.csv') as lang_file:
    langs = [lang.lower() for lang in lang_file.read().splitlines()]
with open('./consts/server.csv') as server_file:
    servers = [server.lower() for server in server_file.read().splitlines()]
with open('./consts/gateway.csv') as gate_file:
    gates = [gate.lower() for gate in gate_file.read().splitlines()]
with open('./consts/monitor.csv') as monitor_file:
    monitors = [monitor.lower() for monitor in monitor_file.read().splitlines()]
with open('./consts/discovery.csv') as disco_file:
    discos = [disco.lower() for disco in disco_file.read().splitlines()]

DATA = {
    'dbs' : dbs, 'servers' : servers, 'buses' : buses, 'langs' : langs, 'gates' : gates, 'monitors' : monitors, 'discos' : discos
}

def are_similar(name, candidate):
    return name == candidate

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
    parts = full_repo_name.split('/')
    if len(parts) != 2:
        return None
    username, repo_name = full_repo_name.split('/')
    workdir = path.join("temp", username)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    full_workdir = path.join(workdir, repo_name)
    if not path.exists(full_workdir):
        print('-cloning repo')
        endpoint = 'https://api.github.com/repos/%s' % (full_repo_name,)
        p1 = subprocess.run(['curl', endpoint], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,)
        data = json.loads(p1.stdout.decode("utf-8"))
        if 'size' not in data or data['size'] < 512000:
            try:
                # force SSH protocol
                repo_url = repo_url.replace("git://github.com/", "git@github.com:")
                print("--repo_url", repo_url)
                git.Git(workdir).clone(repo_url)
            except Exception as e:
                print("cloning repo exception", e)
        else:
            print('repo too big')
            return None
    else:
        print('repo already cloned')
    return full_workdir

def locate_files(workdir, filename):
    print('-locating ', filename)
    res = []
    try:
        for df in Path(workdir).rglob(filename):
            if not df.is_file():
                continue
            df = str(df)
            res.append(df.split(workdir)[-1])
    except OSError:
        pass
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
    result = subprocess.run(['github-linguist', workdir, "--json"], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    dict_langs = json.loads(output)
    languages_list = [lang.lower() for lang in dict_langs if float(dict_langs[lang]['percentage']) > 10]
    return languages_list

def analyze_dockerfile(workdir, df):
    print('-analyzing dockerfile', df)
    analysis = {'path': df, 'cmd': '', 'cmd_keywords': [], 'from' : ''}
    try:
        commands = dockerfile.parse_file(workdir+df)
        runs = ''
        for command in commands:
            if command.cmd == 'from' and command.value:
                analysis['from'] = command.value[0].split(':')[0]
                analysis['from_full'] = command.value[0]
            if command.cmd == 'run':
                runs += '%s ' % (' '.join(command.value),)
            if command.cmd == 'cmd':
                analysis['cmd'] = ' '.join(command.value)
                analysis['cmd_keywords'] = keywords(analysis['cmd'])
            analysis['keywords'] = keywords(runs)
        if 'from' in analysis:
            for k,v in DATA.items():
                analysis[k] = match_one(analysis['from'], v) \
                              or match_ones(get_words(analysis['from']), v) \
                              or match_ones(get_words(analysis['cmd']), v) \
                              or match_ones(get_words(runs), v)
    except dockerfile.GoParseError as e:
        print(e)
    return analysis

def analyze_file(workdir, f):
    print('-analyzing file', f)
    analysis = {'path': f}
    try:
        with open(workdir+f) as fl:
            data = ' '.join(fl.read().splitlines())
            for k,v in DATA.items():
                if k == 'langs':
                    continue
                analysis[k] = match_alls(get_words(data), v)
    except UnicodeDecodeError as e:
        print(e)
    return analysis

def check_shared_db(analysis):
    db_services = set(analysis['detected_dbs']['services'])
    dependencies = []
    for service in analysis['services']:
        dependencies += set(service['depends_on']) & db_services
    return len(set(dependencies)) != len(dependencies)

def committers(workdir):
    try:
        result = subprocess.run(['git', '--git-dir', os.path.join(workdir, '.git'), 'shortlog', '-s'], stdout=subprocess.PIPE, timeout=5)
        output = result.stdout.decode("utf-8")
        return len(output.splitlines())
    except:
        return 0

def analyze_docker_compose(workdir, dc):
    print('-analyzing docker-compose')
    dep_graphs = {'full': nx.DiGraph(), 'micro': None}
    nodes_not_microservice = []
    analysis = {'path': dc, 'num_services': 0, 'services': [], 'detected_dbs': { 'num' : 0, 'names': [], 'services': [], 'shared_dbs' : False} }
    with open(workdir+dc) as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
            services = []
            detected_dbs = []
            if not data or 'services' not in data or not data['services']:
                return analysis
            for name, service in data['services'].items():
                if not service:
                    continue
                s = {}
                s['name'] = name
                if 'image' in service and service['image']:
                    s['image'] =  service['image'].split(':')[0]
                    s['image_full'] =  service['image']
                elif 'build' in service and service['build']:
                    s['image'] = s['image_full'] = service['build']
                else:
                    s['image'] = s['image_full'] =  ''
                if isinstance(s['image'], dict):
                    s['image'] = s['image_full'] =  str(list(s['image'].values())[0])

                for k,v in DATA.items():
                    if k == 'langs':
                        continue
                    s[k] = match_ones(get_words(s['image']), v)

                if s['dbs']:
                    detected_dbs.append({'service' : name, 'name': s['dbs'][0]})

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

                # add the node to the dependencies graph
                dep_graphs['full'].add_node(name)
                # add the edges to the dependencies graph
                dep_graphs['full'].add_edges_from([(name, serv) for serv in s['depends_on']])
                # append the node to the nodes_not_microservice list if the node is not a microservice
                if s['dbs'] or s['servers'] or s['buses'] or s['gates'] or s['monitors'] or s['discos']:
                    nodes_not_microservice.append(name)
            analysis['services'] = services
            analysis['num_services'] = len(services)
            analysis['detected_dbs'] = {'num': len(detected_dbs), \
                                        'names' : list({db['name'] for db in detected_dbs}), \
                                        'services' : [db['service'] for db in detected_dbs]}
            analysis['detected_dbs']['shared_dbs'] = check_shared_db(analysis)

            # copy the full graph
            dep_graphs['micro'] = dep_graphs['full'].copy()
            # delete the not-microservice nodes from the micro dependencies graph
            for node in nodes_not_microservice:
                dep_graphs['micro'].remove_node(node)
            for g in dep_graphs:
                analysis['dep_graph_' + g] = {'nodes': dep_graphs[g].number_of_nodes(),
                                              'edges': dep_graphs[g].number_of_edges(),
                                              'avg_deps_per_service': sum([out_deg for name, out_deg in dep_graphs[g].out_degree]) / dep_graphs[g].number_of_nodes(),
                                              'acyclic': nx.is_directed_acyclic_graph(dep_graphs[g]),
                                              'longest_path': nx.dag_longest_path_length(dep_graphs[g])}

        except (UnicodeDecodeError, yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            print(e)

    return analysis

def compute_size(workdir):
    try:
        root_directory = Path(workdir)
        return sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file() and '.git' not in f.parts)//1000
    except:
        return 0

def synthetize_data(analysis):
    keys = DATA.keys()

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
    analysis['images'] = list({s['from']for s in analysis['dockers'] if s['from']})
    for db in set(analysis['dbs']):
        if 'db' == db[-2:]:
            analysis['dbs'].discard(db)
            analysis['dbs'].add(db[-2:])

    if len(analysis['dbs']) > 1:
        analysis['dbs'].discard('db')
    if len(analysis['gates']) > 1:
        analysis['gates'].discard('gateway')
    if len(analysis['monitors']) > 1:
        analysis['monitors'].discard('monitoring')
    if len(analysis['buses']) > 1:
        analysis['buses'].discard('bus')

    for k in keys:
        analysis['num_%s' % (k,)] = len(analysis[k])
        analysis[k] = list(analysis[k])
    analysis['num_dockers'] = len(analysis['dockers'])
    analysis['num_files'] = analysis['num_dockers'] + len(analysis['files']) + 1
    analysis['avg_size_service'] = analysis['size'] / max(analysis['num_dockers'], 1)


def analyze_repo(url):
    lockfile = "temp/%s.lock" % (''.join(get_words(url)),)
    lock = FileLock(lockfile, timeout=0.01)
    workdir = None
    try:
        with lock:
            analysis = {'url' : url}
            analysis['name'] = url.split('.git')[0].split('git://github.com/')[-1]
            print('analyzing', analysis['name'])
            outfile = path.join('results', analysis['name'].replace('/', '#'))
            outfile = "%s.json" % (outfile,)
            if not path.exists(outfile):
                workdir = clone(url, analysis['name'])
                if not workdir:
                    return
                analysis['commiters'] = committers(workdir)
                analysis['size']=compute_size(workdir)
                analysis['languages'] = analyze_languages(workdir)
                print("Language analysis completed")
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
                fs += locate_files(workdir, 'package.json')

                file_analysis = []
                for f in fs:
                    file_analysis.append(analyze_file(workdir, f))
                analysis['files'] =  file_analysis
                synthetize_data(analysis)
                with open(outfile, 'w', encoding='utf-8') as f:
                    analysis = remove_invalid_char(analysis)
                    json.dump(analysis, f, ensure_ascii=False, indent=4)
                shutil.rmtree(path.dirname(workdir))
            else:
                print('skipped')
    except Timeout:
        print('in progress')
    except FileNotFoundError:
        print('FileNotFoundError skipped')
    except Exception as e:
        print('Error, continuing...', e)
    finally:
        print(workdir)


def remove_invalid_char(d):
    if isinstance(d, str):
        return d.encode('utf-16', 'surrogatepass').decode('utf-16')
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = remove_invalid_char(v)
    elif isinstance(d, list) or isinstance(d, set) or isinstance(d, tuple):
        for i, v in enumerate(list(d)):
            d[i] = remove_invalid_char(v)
    return d



def analyze_all():
    repos = Path('repos').glob('*.csv')
    repos = sorted([str(x) for x in repos])
    for source in repos:
        with open(str(source), newline='') as f:
            reader = csv.reader(f, delimiter=',')
            while True:
                try:
                    line = next(reader)
                    print(line, source)
                    analyze_repo(line[0])
                except UnicodeDecodeError as e:
                    print(e)
                except StopIteration:
                    break

analyze_all()