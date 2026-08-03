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
from pipeline3_NC import run_all_NC

from utils import save_results_NC

with open("config_NC.yaml") as f:
    config_NC = yaml.safe_load(f)

results_NC = run_all_NC(config_NC)
save_results_NC(results_NC)
