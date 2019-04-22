import requests

# you need to modify this to your zenodo ID
zenodo_id = 2648592

r = requests.get("https://zenodo.org/api/records/%i" % zenodo_id)

file = open("zenodo_filelist.txt", "w")

for f in r.json()["files"]:
    file.write(f["links"]["self"] + "\n")

file.close()
