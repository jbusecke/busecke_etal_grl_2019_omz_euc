#!/bin/bash

# Create filelist !!!YOU HAVE TO CHANGE YOUR Repository ID
python scripts/parse_filenames.py

#download the files in the list
cd data
wget -i ../zenodo_filelist.txt

#delete the filelist
rm ../zenodo_filelist.txt
