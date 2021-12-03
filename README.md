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
- ```python analyze_results.py``` with optional parameters ```-f include.csv -s 5```
	- ```-f``` the filter file containing the manually filtered repositories
	- ```-s``` include only the repositories with this minimum number of microservices