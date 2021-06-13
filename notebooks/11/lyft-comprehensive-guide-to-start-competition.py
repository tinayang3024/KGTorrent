#!/usr/bin/env python
# coding: utf-8

# # Lyft: Comprehensive guide to start competition
# 
# ![](http://www.l5kit.org/_images/av.jpg)
# <cite>The image from L5Kit official document: <a href="http://www.l5kit.org/README.html">http://www.l5kit.org/README.html</a></cite>
# 
# In this kernel, I will explain how to setup develop environment & the data for this competition.<br/>
# 
# The dataset structure is a bit complicated since the Lyft Level 5 dataset contains various kinds of information.
# Lyft provided a useful library to deal with this data, we need to learn how to fully-utilize these provided tools at first!

# # Competition description
# 
#  - Official page: [https://self-driving.lyft.com/level5/prediction/](https://self-driving.lyft.com/level5/prediction/)
# 
# <blockquote>
#     The dataset consists of 170,000 scenes capturing the environment around the autonomous vehicle. Each scene encodes the state of the vehicle’s surroundings at a given point in time.
# </blockquote>
# 
# <div style="clear:both;display:table">
# <img src="https://self-driving.lyft.com/wp-content/uploads/2020/06/motion_dataset_lrg_redux.gif" style="width:45%;float:left"/>
# <img src="https://self-driving.lyft.com/wp-content/uploads/2020/06/motion_dataset_2-1.png" style="width:45%;float:left"/>
# </div>
# 
# <br/>
# <p><b>The goal of this competition is to predict other car/cyclist/pedestrian (called "agent")'s motion.</b><p>
# 
# <img src="https://self-driving.lyft.com/wp-content/uploads/2020/06/diagram-prediction-1.jpg" style="width:70%"/>
# 
#     
# <p><a href="https://vimeo.com/389096888">Lyft Self-Driving Employee Rides Testing</a> from <a href="https://vimeo.com/user99616812">Lyft Level 5</a> on <a href="https://vimeo.com">Vimeo</a>.</p>

# In[1]:


from IPython.display import IFrame
IFrame('https://player.vimeo.com/video/389096888', width=640, height=360, frameborder="0", allow="autoplay; fullscreen", allowfullscreen=True)


# # Environment setup
# 
# Following instruction will work only on kaggle kernel.<br/>
# Please try following this step by creating new kernel (or forking this kernel) on kaggle.
# 
# ## Install the additional library: l5kit
# We want to install `l5kit` library, which is provided by Lyft to handle this competition's dataset (Level5 Prediction Dataset).<br/>
# Here I use the "Utility script" functionality of kaggle kernel, instead of installing the library by `pip install` command.
# 
# 
# Click "File" botton on top-left, and choose "Add utility script". For the pop-up search window, you need to remove "Your Work" filter, and search "[philculliton/kaggle-l5kit](https://www.kaggle.com/mathurinache/kaggle-l5kit)" on top-right of the search window.
# Then you can add the **kaggle-l5kit** utility script.
# 
# If successful, you can see "usr/lib/kaggle-l5kit" is added to the "Data" section of this kernel page on right side of the kernel.
# 
# Reference for utility script
#  - [Feature Launch: Import scripts into notebook kernels](https://www.kaggle.com/product-feedback/91185)
#  - [Import functions from Kaggle script](https://www.kaggle.com/rtatman/import-functions-from-kaggle-script)

# In[2]:


# Running this pip install code takes time, we can skip it when we attach utility script correctly!
# !pip install -U l5kit


# In[3]:


import l5kit

l5kit.__version__


# As we can see, we don't need to hassle with the time to run `pip install` which takes time to install the library. Just attach the utility script and the library is already setup!

# ## Attach additional dataset source: for config files
# 
# We are using some config files when we want to load/visualize Lyft level5 dataset.<br/>
# @jpbremer already uploaded config files as Kaggle Dataset platform: [lyft-config-files](https://www.kaggle.com/jpbremer/lyft-config-files).<br/>
# This is originally from official github [lyft/l5kit example page](https://github.com/lyft/l5kit/tree/master/examples).
# 
# Click "Add data" button and press "Search by URL". Typing "https://www.kaggle.com/jpbremer/lyft-config-files" shows the dataset.<br/>
# Once the dataset is successfully added you can see "lyft-config-files" as dataset on left side bar.

# ## import

# In[4]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# In[5]:


from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os

from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')


# In[6]:


# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    filenames.sort()
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Basic tutorial
# 
# Before going to look the data, we still need to learn some advanced functionality to handle data.
# 
# This section basically follows [Data format](https://github.com/lyft/l5kit/blob/master/data_format.md) description page of l5kit.
# 
# ## numpy structured array
# 
# Lyft dataset uses numpy's [structured array](https://docs.scipy.org/doc/numpy/user/basics.rec.html) functionality to store various kinds of features together.<br/>
# Let's see example to how it works

# In[7]:


my_arr = np.zeros(3, dtype=[("color", (np.uint8, 3)), ("label", np.bool)])

my_arr[0]["color"] = [0, 218, 130]
my_arr[0]["label"] = True
my_arr[1]["color"] = [245, 59, 255]
my_arr[1]["label"] = True

my_arr


# At top line we defined length=3 array. Usually each element of array consists of only integer or float, but we can define custom structured format by specifying `dtype` as list of "fields" which consists of name and structure.
# 
# Above example contains 2 fields. 1. 8byte uint with length 3 array, 2. single element boolean array.
# 
# As you can see, `my_arr[i]["name"]` will access i-th element's "name" field.
# 
# Usually when we train neural network, we would like to access all the field of random i-th element.
# According to [Lyft Data Format page](https://github.com/lyft/l5kit/blob/master/data_format.md), "Structured arrays are a great fit to group this data together in memory and on disk.", rather than preparing "color" array and "label" array separately and access each array's i-th element, especially when the number of field glow. 

# ## zarr

# Zarr data format is used to store and read these numpy structured arrays from disk.<br/>
# Zarr allows us to write very large (structured) arrays to disk in n-dimensional compressed chunks.
# 
# Here is a short tutorial:

# In[8]:


import zarr

z = zarr.open("./dataset.zarr", mode="w", shape=(500,), dtype=np.float32, chunks=(100,))

# We can write to it by assigning to it. This gets persisted on disk.
z[0:150] = np.arange(150)


# As we specified chunks to be of size 100, we just wrote to two separate chunks. On your filesystem in the dataset.zarr folder you will now find these two chunks. As we didn't completely fill the second chunk, those missing values will be set to the fill value (defaults to 0). The chunks are actually compressed on disk too! 
# 
# We can print some info: by not doing much work at all we saved almost 75% in disk space!

# In[9]:


print(z.info)


# When we check filesystem, `dataset.zarr` directory is created and there are 2 files "0" and "1" which are chunks currently created by just assigning value to zarr array.

# In[10]:


get_ipython().system('ls -l ./*')


# Reading from a zarr array is as easy as slicing from it like you would any numpy array. The return value is an ordinary numpy array. Zarr takes care of determining which chunks to read from.

# In[11]:


print(z[::20]) # Read every 20th value


# Zarr supports StructuredArrays, the data format we use for our datasets are a set of structured arrays stored in zarr format.
# 
# Some other zarr benefits are:
# 
#  - Safe to use in a multithreading or multiprocessing setup. Reading is entirely safe, for writing there are lock mechanisms built-in.
#  - If you have a dataset that is too large to fit in memory, loading a single sample becomes my_sample = z[sample_index] and you get compression out of the box.
#  - The blosc compressor is so fast that it is faster to read the compressed data and uncompress it than reading the uncompressed data from disk.
#  - Zarr supports multiple backend stores, your data could also live in a zip file, or even a remote server or S3 bucket.
#  - Other libraries such as xarray, Dask and TensorStore have good interoperability with Zarr.
#  - The metadata (e.g. dtype, chunk size, compression type) is stored inside the zarr dataset too. If one day you decide to change your chunk size, you can still read the older datasets without changing any code.

# See the zarr [docs](https://zarr.readthedocs.io/en/stable/) for more details.

# # Lyft's dataset structure
# 
# Now the basics learning are done! We can start looking Lyft level 5 dataset using `l5kit` library.
# 
# I referenced [l5kit visualize_data.ipynb](https://github.com/lyft/l5kit/blob/master/examples/visualisation/visualise_data.ipynb) for this section.
# 
# 
# ## Word definition
#  - **"Ego"** is the host car which is recording/measuring the dataset.
#  - **"Agent"** is the surronding car except "Ego" car.
#  - **"Frame"** is the 1 image snapshot, where **"Scene"** is made of multiple frames of contious-time (video).
# 
# ## Class diagram
# <img src="https://storage.googleapis.com/kaggle-forum-message-attachments/987047/16744/l5kit_class.png" width="600" />
# 
# <br/>
# 
# ## Initial setup

# In[12]:


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
# get config
cfg = load_config_data("/kaggle/input/lyft-config-files/visualisation_config.yaml")
print(cfg)


# ## Loading data
# 
# Here we will only use the first dataset from the sample set.
# 
# We're building a `LocalDataManager` object. This will resolve relative paths from the config using the `L5KIT_DATA_FOLDER` env variable we have just set.
# 
# Here sample.zarr data is used for visualization, please use train.zarr / validate.zarr / test.zarr for actual model training/validation/prediction.
# 

# In[13]:


dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)


# In[14]:


dataset_path


# ## Working with the raw data
# 
# `zarr_dataset` contains **scenes, frames, agents, tl_faces** attributes, which are the raw structured array data.
# 
# Each data structure definition can be checked at [here](https://github.com/lyft/l5kit/blob/master/data_format.md#2020-lyft-competition-dataset-format).
# 
# ### scenes
# 
# ```
# SCENE_DTYPE = [
#     ("frame_index_interval", np.int64, (2,)),
#     ("host", "<U16"),  # Unicode string up to 16 chars
#     ("start_time", np.int64),
#     ("end_time", np.int64),
# ]
# ```
# 
# ### frames
# 
# ```
# FRAME_DTYPE = [
#     ("timestamp", np.int64),
#     ("agent_index_interval", np.int64, (2,)),
#     ("traffic_light_faces_index_interval", np.int64, (2,)),
#     ("ego_translation", np.float64, (3,)),
#     ("ego_rotation", np.float64, (3, 3)),
# ]
# ```
# 
# ### agents
# 
# ```
# AGENT_DTYPE = [
#     ("centroid", np.float64, (2,)),
#     ("extent", np.float32, (3,)),
#     ("yaw", np.float32),
#     ("velocity", np.float32, (2,)),
#     ("track_id", np.uint64),
#     ("label_probabilities", np.float32, (len(LABELS),)),
# ]
# ```
# 
# ### traffic_light_faces
# 
# ```
# TL_FACE_DTYPE = [
#     ("face_id", "<U16"),
#     ("traffic_light_id", "<U16"),
#     ("traffic_light_face_status", np.float32, (len(TL_FACE_LABELS,))),
# ]
# ```

# As an example, we will try scatter plot using **frames "ego_translation"** data. This is the movement of ego car.

# In[15]:


frames = zarr_dataset.frames

## This is slow.
# coords = np.zeros((len(frames), 2))
# for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):
#     frame = zarr_dataset.frames[idx_data]
#     coords[idx_coord] = frame["ego_translation"][:2]

# This is much faster!
coords = frames["ego_translation"][:, :2]

plt.scatter(coords[:, 0], coords[:, 1], marker='.')
axes = plt.gca()
axes.set_xlim([-2500, 1600])
axes.set_ylim([-2500, 1600])
plt.title("ego_translation of frames")


# [Note] Performance-aware slicing
# 
# I commented out some codes which uses for loop. This is slow because when we call `zarr_dataset.frames[idx_data]`, **data is decompressed everytime** using zarr format.<br/>
# Instead, we can remove for loop by using slice accessing. `frames["ego_translation"]` is same as `frames[:]["ego_translation"]`, which accesses all the element's "ego_translation" field. By writing this, number of decompression call is reduced and code runs faster dramatically.

# ## pytorch Dataset class
# 
# Instead of working with raw data, L5Kit provides PyTorch ready datasets.
# It's much easier to use this wrapped dataset class to access data.
# 
# 2 dataset class is implemented.
# 
#  - **EgoDataset**: this dataset iterates over the AV (Autonomous Vehicle) annotations
#  - **AgentDataset**: this dataset iterates over other agents annotations

# In[16]:


# 'map_type': 'py_semantic' for cfg.
semantic_rasterizer = build_rasterizer(cfg, dm)
semantic_dataset = EgoDataset(cfg, zarr_dataset, semantic_rasterizer)


# Rasterizer is in charge of visualizing the data, as we will see next.

# ## Visualization example
# 
# Lyft l5kit also provides visualization functionalities.<br/>
# We will visualize data to understand what kind of information is stored in this dataset.<br/>

# We can sample a data from dataset, and convert to RGB image using `rasterizer`.
# 
#  - image: (channel, height, width) image of a frame. This is Birds-eye-view (BEV) representation.
#  - target_positions: (n_frames, 2) displacements in meters in world coordinates
#  - target_yaws: (n_frames, 1)
#  - centroid: (2) center position x&y.
#  - world_to_image: (3, 3) 3x3 matrix, used for transform matrix.
# 
# Data is represented as 2.5D, positions and yaws are separated.<br/>
# target_positions are represented in world coordinates and it is converted to pixel coordinates to visualize below:

# In[17]:


def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    plt.title(title)
    plt.imshow(im[::-1])
    plt.show()

visualize_trajectory(semantic_dataset, index=0)


# We can switch rasterizer to visualize satellite image easily!

# In[18]:


# map_type was changed from 'py_semantic' to 'py_satellite'.
cfg["raster_params"]["map_type"] = "py_satellite"
satellite_rasterizer = build_rasterizer(cfg, dm)
satellite_dataset = EgoDataset(cfg, zarr_dataset, satellite_rasterizer)

visualize_trajectory(satellite_dataset, index=0)


# In[19]:


type(satellite_rasterizer), type(semantic_rasterizer)


# Now we visualized **EgoDataset**.
# 
# **AgentDataset** can be used to visualize an agent. This dataset iterates over agents and not the AV anymore, and the first one happens to be the pace car (you will see this one around a lot in the dataset).

# In[20]:


agent_dataset = AgentDataset(cfg, zarr_dataset, satellite_rasterizer)
visualize_trajectory(agent_dataset, index=0)


# You can compare above 2 images, only the target car changed from host to agent.

# <blockquote> <b>System Origin and Orientation</b>
# 
# At this point you may have noticed that we flip the image on the Y-axis before plotting it.
# 
# When moving from 3D to 2D we stick to a right-hand system, where the origin is in the bottom-left corner with positive x-values going right and positive y-values going up the image plane. The camera is facing down the negative z axis.
# 
# However, both opencv and pyplot place the origin in the top-left corner with positive x going right and positive y going down in the image plane. The camera is facing down the positive z-axis.
# 
# The flip done on the resulting image is for visualisation purposes to accommodate the difference in the two coordinate frames.
# 
# Further, all our rotations are counter-clockwise for positive value of the angle.
# </blockquote>
# 

# ## Scene handling
# 
# Both EgoDataset and AgentDataset provide 2 methods for getting interesting indices:
# 
#  - **get_frame_indices** returns the indices for a given frame. For the `EgoDataset` this matches a single observation, while more than one index could be available for the `AgentDataset`, as that given frame may contain more than one valid agent
#  - **get_scene_indices** returns indices for a given scene. For both datasets, these might return more than one index

# In[21]:


from IPython.display import display, clear_output
import PIL
 
dataset = semantic_dataset
scene_idx = 34
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    images.append(PIL.Image.fromarray(im[::-1]))


# In[22]:


get_ipython().run_cell_magic('capture', '', '# From https://www.kaggle.com/jpbremer/lyft-scene-visualisations by @jpbremer\ndef animate_solution(images):\n\n    def animate(i):\n        im.set_data(images[i])\n        return (im,)\n \n    fig, ax = plt.subplots()\n    im = ax.imshow(images[0])\n    def init():\n        im.set_data(images[0])\n        return (im,)\n    \n    return animation.FuncAnimation(fig, animate, init_func=init, frames=len(images), interval=60, blit=True)\n\nanim = animate_solution(images)')


# In[23]:


HTML(anim.to_jshtml())


# # Next to go
# 
# That's all for first introduction to familialize with l5kit dataset.<br/>
# We have still need to know a lot to write data processing code to build a better prediction model pipeline.<br/>
# I wrote next kernel [Lyft: Deep into the l5kit library](https://www.kaggle.com/corochann/lyft-deep-into-the-l5kit-library) for this purpose.
# 
# If you don't want quickly try training prediction model baseline, you can go to [agent motion prediction ipynb](https://github.com/lyft/l5kit/tree/master/examples/agent_motion_prediction) to start prediction.<br/>
# The same content is already uploaded: [a port of l5kit example](https://www.kaggle.com/hirune924/just-a-port-of-l5kit-example/data) by @hirune924.
# 
# **[Update] I also wrote training tutorial kernel:**
#  - [Lyft: Training with multi-mode confidence](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence)
#  - [Lyft: Prediction with multi-mode confidence](https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence)

# # Further reference
# 
#  - Paper of this Lyft Level 5 prediction dataset: [One Thousand and One Hours: Self-driving Motion Prediction Dataset](https://arxiv.org/abs/2006.14480)
#  - [jpbremer/lyft-scene-visualisations](https://www.kaggle.com/jpbremer/lyft-scene-visualisations)

# <h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>
