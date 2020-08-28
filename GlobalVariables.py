import numpy as np
import pandas as pd
import warnings
from fredapi import Fred

#TODO move below into defaultModelKwargs and update any functions accordingly
stdTestingRange = np.logspace(-3, 2, 100)

# File logistics:
fileLocation = "D:\\Users\\dzjre\PycharmProjects\FinModelingResearchSummer2020\Data"
fred = Fred(api_key='93a721eba3644b125df12f223484e552')

# Format settings:
outputRJust = 70
brk = "\n\n\n\n"*2
sBrk = "\n\n"

# Printing settings:
extraDetails = True  # Additional printouts/graphs for testing
tempThing = False # Remove
debug = False
graph = False
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Used to ignore np.polyfit rankwarning error when handling 2 asset portfolios:
warnings.simplefilter('ignore', np.RankWarning)
