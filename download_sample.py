import urllib.request
import os

url = "https://www.wavsource.com/snds_2020-10-01_3728627494378403/people/women/about_time_f.wav"
out_path = "/Users/alyasi/apva/data/ref/female_soft.wav"

try:
    print(f"Downloading {url} to {out_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Download successful!")
except Exception as e:
    print(f"Error: {e}")
