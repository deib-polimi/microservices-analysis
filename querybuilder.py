import datetime
from github import Github 
from time import sleep


def date_to_string(date):
    return date.strftime('%Y-%m-%d')

query = "microservice filename:Dockerfile"
startdate_str = '2015-01-01'
startdate = datetime.datetime.strptime(startdate_str, '%Y-%m-%d')
print(startdate)
enddate = datetime.date.today()

#d1 = today.


minsize = l = 0
maxsize = r = 10**10



token = '9f35bc1eb979963e252d5a1651f17a2656cdca0d'
g = Github(token)
req = g.search_code(query=f'{query} size:{l}..{r}')
res = set()
while req.totalCount >= 1000:
    r = r//2
    sleep(3)
    req = g.search_code(query=f'{query} size:{l}..{r}')
    tot = req.totalCount
    while tot >= 1000:
        r = r//2
        req = g.search_code(query=f'{query} size:{l}..{r}')
        sleep(3)
        tot = req.totalCount
        print(l,r,tot)
    res.add((l, r))
    l, r = r, maxsize
    req = g.search_code(query=f'{query} size:{l}..{r}')


res.add((l, r))

print(res)