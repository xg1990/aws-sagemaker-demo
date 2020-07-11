import os
import pathlib
import json


CONFIG_FN = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "cloud_config.json"
)

with open(CONFIG_FN) as fp:
    CLOUD_CONFIG = json.load(fp)