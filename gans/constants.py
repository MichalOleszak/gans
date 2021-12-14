import os
from pathlib import Path


PROJECT_ROOT_ABS_DIRPATH = Path(__file__).parent.parent
DATA_DIRPATH = os.path.join(PROJECT_ROOT_ABS_DIRPATH, "data")
POKEMON_DATA_DIRPATH = os.path.join(DATA_DIRPATH, "pokemon")
