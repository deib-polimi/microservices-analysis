from pathlib import Path
import json


repos = Path('results').glob('*.json')
i = j = 0
output_repos = []
for source in repos:
    j += 1
    try:
        with open(str(source)) as json_file:
            data = json.load(json_file)
            if data['url']:
                output_repos.append({'url': data['url'], 'name': data['name']})
    except (UnicodeDecodeError, json.decoder.JSONDecodeError):
        i += 1
        pass

with open("repos.csv", "w") as output_file:
    for repo in output_repos:
        output_file.write("github.com/" + repo['name'] + ";" + repo['url'] + "\n")

print(f"Results repo: {j}, errors:{i}, written to output file: {len(output_repos)}")