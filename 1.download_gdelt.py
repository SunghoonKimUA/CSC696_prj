import requests
from zipfile import ZipFile
from datetime import date
from datetime import timedelta

url = 'http://data.gdeltproject.org/events/'
gdelt_path = './data\gdelt_1_0/'

def download_unzip(file_name):
    r = requests.get(url+file_name, allow_redirects=True)
    open(gdelt_path+file_name, 'wb').write(r.content)
    with ZipFile(gdelt_path+file_name, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall(path=gdelt_path)

"""
for yyyy in range(2000, 2014):
    mm = ''
    dd = ''
    if yyyy > 2005:
        for mm in range(1, 13):
            if str(yyyy)+str(mm).zfill(2) <= '201303':
                print(str(yyyy)+str(mm).zfill(2))
                download_unzip(str(yyyy)+str(mm).zfill(2)+".zip")
    else:
        print(yyyy)
        download_unzip(str(yyyy)+".zip")
"""

# d = date(2013, 4, 1)
d = date(2020, 8, 28)
day = timedelta(days=1)
for day_gap in range(1, 365*9):
    if d < date(2022, 3, 31):
        print(d.strftime('%Y%m%d'))
        download_unzip(d.strftime('%Y%m%d')+".export.CSV.zip")
        d = d + day

print("End")