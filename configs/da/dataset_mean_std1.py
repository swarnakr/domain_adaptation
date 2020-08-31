

"""

https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/15

https://stackoverflow.com/questions/895929/how-do-i-determine-the-standard-deviation-stddev-of-a-set-of-values/897463#897463

Welford's method for accurately computing running variance: https://www.johndcook.com/blog/standard_deviation/

"""

#from datasets import build_dataset
from data import spacenet
from data import make_data_loader
from common import config
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_mean_variance(dataloader):

    pixel_mean = np.zeros(3)

    pixel_std = np.zeros(3)

    k = 1

    tbar =  tqdm(dataloader)
    for i, sample in enumerate(tbar):
        image  = sample['image']

            
        image = np.array(image).transpose([0,2,3,1])

        pixels = image.reshape((-1, image.shape[3]))

        for pixel in pixels:

            diff = pixel - pixel_mean

            pixel_mean += diff / k

            pixel_std += diff * (pixel - pixel_mean)

            k += 1

    # pixel number is k-1

    pixel_std = np.sqrt(pixel_std / (k - 2))

    return pixel_mean, pixel_std

if __name__ == '__main__':

    city = "Shanghai"

    dataloader, _, _, _ = make_data_loader(config)
    mean, std = compute_mean_variance(dataloader)

    print("mean: {}\nstd: {}".format(mean, std))

