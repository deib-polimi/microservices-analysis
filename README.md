# microservices-analysis

### Setup
- Requirements:
	- Ruby
	- github-linguist: ```gem install github-linguist```

### Execution
Copy the resulting file from the Crawler to the ```repos``` folder, then run:
- ```python analyze_repo.py```

Output the repo links to a ``repos.csv`` file with:
- ``python output_repo.py``

After the manual filtering, run:
- ```python analyze_results.py -f include.csv``` where ```-f``` is the filter file