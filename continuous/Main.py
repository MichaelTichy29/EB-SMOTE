# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd





####################################
#####     Statistical results      #######
####################################






####################################
#####  Main        #######
####################################


import yaml
#from pipeline1 import run_all
from pipeline3 import run_all

from utils import save_results

with open("config.yaml") as f:
    config = yaml.safe_load(f)

results = run_all(config)
save_results(results)
