import os

import imageio
from torch.utils.data import Dataset
import torchvision.transforms as T

from gans import constants


def get_basic_filepaths():
    filenames = [x for x in os.listdir(constants.POKEMON_DATA_DIRPATH) if x.strip(".png").isdecimal()]
    return [os.path.join(constants.POKEMON_DATA_DIRPATH, f)
            for f in filenames
            if imageio.imread(os.path.join(constants.POKEMON_DATA_DIRPATH, f)).shape == (96, 96, 4)]


set_to_func = {
    "basic": get_basic_filepaths,
}


class PokemonDataset(Dataset):
    def __init__(self, set="basic"):
        self.set = set
        self.fps = set_to_func[set]()

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        image = imageio.imread(self.fps[index])
        X = self.transform(image)
        return X

    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
