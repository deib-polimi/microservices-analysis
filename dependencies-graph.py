import argparse
import yaml
import networkx as nx
from matplotlib import pyplot as plt

def analyze_docker_compose(workdir, dc):
    print('-analyzing docker-compose')
    dep_graph = nx.DiGraph()
    analysis = {'path': dc, 'num_services': 0, 'services': [], 'detected_dbs': { 'num' : 0, 'names': [], 'services': [], 'shared_dbs' : False} }
    with open(workdir+dc) as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
            services = []

            #print(data["services"].items())
            if not data or 'services' not in data or not data['services']:
                return analysis
            for name, service in data['services'].items():
                if not service:
                    continue
                s = {}

                if 'depends_on' in service:
                    if isinstance(service['depends_on'], dict):
                        s['depends_on'] = list(service['depends_on'].keys())
                    else:
                        s['depends_on'] = service['depends_on']
                elif 'links' in service:
                    print( list(service['links']))
                    s['depends_on'] = list(service['links'])
                else:
                    s['depends_on'] = []
                dep_graph.add_edges_from([(name, service) for service in s['depends_on']])
                services.append(s)
            analysis['services'] = services
            analysis['num_services'] = len(services)

        except (UnicodeDecodeError, yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            print(e)

    print("in degree", dep_graph.in_degree)
    print("out degree", dep_graph.out_degree)
    print("avg deps per service", sum([out_deg for name, out_deg in dep_graph.out_degree])/dep_graph.number_of_nodes())
    print("acyclic?", nx.is_directed_acyclic_graph(dep_graph))
    print("longest path length?", nx.dag_longest_path_length(dep_graph))
    print("topological sorting", list(nx.topological_sort(dep_graph)))
    plt.tight_layout()
    nx.draw_networkx(dep_graph, arrows=True)
    plt.savefig("g2.png", format="PNG")
    plt.clf()
    return analysis

parser = argparse.ArgumentParser()
parser.add_argument("-f", dest='file', type=str, help="docker-compose.yml file", required=True)
args = parser.parse_args()

analysis = analyze_docker_compose("./docker-compose-test/", args.file)
print(analysis)