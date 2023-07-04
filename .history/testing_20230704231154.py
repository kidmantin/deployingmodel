import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import bentoml
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

spectrogram = pd.read_csv(r"C:\Users\Kidma\source\repos\deployingmodel\large_df.csv").to_numpy()
spectrogram = np.expand_dims(spectrogram, 2)