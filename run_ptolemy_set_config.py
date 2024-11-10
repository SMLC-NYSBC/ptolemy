# first assuming we've run ptolemy by uvicorn ptolemy.ptolemy_server:app --host 192.168.33.26 --port 80001

from ptolemy.mrc import load_mrc
import numpy as np
from ptolemy.Requester import Ptolemy_Requester
import pandas as pd

requester = Ptolemy_Requester('http://192.168.33.26:8001/')

# make sure the gpu here is available
requester.set_config('/h2/pkim/ptolemy/ptolemy/241022_ptolemy_config_4cut_shapefilter.json')
