import pdb
from common import config
import os
import json

def get_updated_filenames(city, split):
    name_root = '/home/home1/swarnakr/scriptln/DA/domains/' + city; 
    with open(os.path.join(name_root, split + '.json')) as f:
        files = json.load(f)

    new_files=[]
    for ff in files:
        paths = [os.path.join(config.img_root, '{}_out_{}.png'.format(ff,ii+1)) for ii in range(5)]
        for pp in paths:
            if os.path.exists(pp):
                ind = pp.rfind('/')
                new_files.append(pp[ind+1:])


    save_name = os.path.join('/home/home1/swarnakr/scriptln/DA/domains/',city, '{}_transformed.json'.format(split))
    print(len(new_files))
    with open(save_name,'w') as f:
        json.dump(new_files,f)
                             

def main():
    #get_updated_filenames('Shanghai','train')
    get_updated_filenames('Shanghai','val')

if __name__ == "__main__":
    main()
