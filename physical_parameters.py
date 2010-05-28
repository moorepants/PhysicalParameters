import tarfile
import os

# uncompress the data file
tar = tarfile.open('data.tar.gz')
tar.extractall()
tar.close()

# create data directory and move the pendat folder inside
os.system('mkdir data')
os.system('mv pendDat data/pendDat')
os.system('mv data.mat data/data.mat')
os.system('mv MeasUncert.txt data/MeasUncert.txt')

# run the scripts to process the data
os.system('python calc_parameters.py')
os.system('python calc_canon.py')
