import tensorflow as tf
from numpy.random import seed
import numpy as np
import os 
import yaml
import logging

import geopandas as gpd
import streamlit as st

# log messages to a file
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filemode='w', filename="dl.log", level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
logging.info("Im about to eat pizza(s) when...")


def load_streets(file):
        streets_df = gpd.read_file(file, crs="EPSG:4326")
        return streets_df

# Reproducability
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    logging.info("-- set seed")
    
