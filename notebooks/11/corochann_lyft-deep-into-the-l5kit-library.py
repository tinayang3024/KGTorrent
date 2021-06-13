#!/usr/bin/env python
# coding: utf-8

# # Lyft: Deep into the l5kit library
# 
# ![](http://www.l5kit.org/_images/av.jpg)
# <cite>The image from L5Kit official document: <a href="http://www.l5kit.org/README.html">http://www.l5kit.org/README.html</a></cite>
# 
# Continued from the previous kernel [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition).
# 
# In this kernel, I will look into the **raw data structures** and **l5kit library** in more detail **with code reading**. After understanding these, I hope you can arrange the data by yourself to build a better pipleline for motion prediction.
# 
# 
# # Table of Contents
# 
# ** [1. Understanding Rasterizer class](#rasterizer)** <br>
# ** [2. Understanding EgoDataset/AgentDataset class](#ego_agent_dataset)** <br>
# ** [3. Understanding raw data structures](#raw_data)** <br>
# 
# 
# The first part is same with previous kernel, please jump to [1. Understanding Rasterizer class](#rasterizer) for the main topic of this kernel.

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

# # Environment setup
# 
#  - Please add "[philculliton/kaggle-l5kit](https://www.kaggle.com/mathurinache/kaggle-l5kit)" as utility script
#  - Please add [lyft-config-files](https://www.kaggle.com/jpbremer/lyft-config-files) as dataset
#  
# See previous kernel [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition) for details.

# ## import

# In[ ]:


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


# In[ ]:


import zarr

import l5kit
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

from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)


# In[ ]:


from IPython.display import display, clear_output
import PIL


# Originally from https://www.kaggle.com/jpbremer/lyft-scene-visualisations by @jpbremer
# Modified following:
#  - Added to show timestamp
#  - Do not show image, to only show animation.
#  - Use blit=True.

def animate_solution(images, timestamps=None):
    def animate(i):
        changed_artifacts = [im]
        im.set_data(images[i])
        if timestamps is not None:
            time_text.set_text(timestamps[i])
            changed_artifacts.append(im)
        return tuple(changed_artifacts)

    
    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    if timestamps is not None:
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=60, blit=True)
    
    # To prevent plotting image inline.
    plt.close()
    return anim


# # Initial setup
# 
# ## Word definition
#  - **"Ego"** is the host car which is recording/measuring the dataset.
#  - **"Agent"** is the surronding car except "Ego" car.
#  - **"Frame"** is the 1 image snapshot, where **"Scene"** is made of multiple frames of contious-time (video).
# 
# ## Class diagram
# <a id="class_diagram"></a>
# <img src="https://storage.googleapis.com/kaggle-forum-message-attachments/987047/16744/l5kit_class.png" width="600" />
# 
# <br/>

# ## Loading data
# 
# Here we will only use the first dataset from the sample set. (sample.zarr data is used for visualization, please use train.zarr / validate.zarr / test.zarr for actual model training/validation/prediction.)<br/>
# We're building a `LocalDataManager` object. This will resolve relative paths from the config using the `L5KIT_DATA_FOLDER` env variable we have just set.

# In[ ]:


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
# get config
cfg = load_config_data("/kaggle/input/lyft-config-files/visualisation_config.yaml")
print(cfg)


# In[ ]:


dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)


# <a id="rasterizer"></a>
# 
# # 1. Understanding Rasterizer class
# 
# ## Rasterizer class
# 
# The first topic I will introduce is "Rasterizer". This class supports 2 methods (See base [Rasterizer](https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/rasterization/rasterizer.py) class).
# 
#  - `rasterize` method: to create (ch, height, width) format image. Basically this can be used for the input of prediciton model. It can have any number of channels.
#  - `to_rgb` method: to convert image made by rasterize method into RGB image (ch=3, height, width).
#  
# `l5kit` already provides several kinds of Rasterizer, each can be instantiated via [build_rasterizer](https://github.com/lyft/l5kit/blob/1e235b8617488e818be30cd7193d43588125bbab/l5kit/l5kit/rasterization/rasterizer_builder.py#L99) method with `cfg`. Let's see each class's role.

# In[ ]:


def visualize_rgb_image(dataset, index, title="", ax=None):
    """Visualizes Rasterizer's RGB image"""
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)

    if ax is None:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.imshow(im[::-1])


# In[ ]:


# Prepare all rasterizer and EgoDataset for each rasterizer
rasterizer_dict = {}
dataset_dict = {}

rasterizer_type_list = ["py_satellite", "satellite_debug", "py_semantic", "semantic_debug", "box_debug", "stub_debug"]

for i, key in enumerate(rasterizer_type_list):
    # print("key", key)
    cfg["raster_params"]["map_type"] = key
    rasterizer_dict[key] = build_rasterizer(cfg, dm)
    dataset_dict[key] = EgoDataset(cfg, zarr_dataset, rasterizer_dict[key])


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, key in enumerate(["stub_debug", "satellite_debug", "semantic_debug", "box_debug", "py_satellite", "py_semantic"]):
    visualize_rgb_image(dataset_dict[key], index=0, title=f"{key}: {type(rasterizer_dict[key]).__name__}", ax=axes[i])
fig.show()


# We see that
#  - `StubRasterizer` is just for debugging, creates all black image with specified (height, width).
#  - `BoxRasterizer` creates Ego (host car) as green box, and Agent as blue box.
#  - `SatelliteRasterizer` draws satellite map.
#  - `SemanticRasterizer` draws semantic map which contains lane & crosswalk information
#  - `SatBoxRasterizer` = SatelliteRasterizer + BoxRasterizer
#  - `SemBoxRasterizer` = SemanticRasterizer + BoxRasterizer
# 
# 
# Note that I guess Satellite image is NOT taken at the same time when host car moves, so the car on satellite image does NOT match with the car on drawn in BoxRasterizer!<br/>
# Satellite image is useful to get detailed information about the current place, but does not represent the current situation of car or traffic light etc.

# ## Meaning of color in semantic map?
# 
# Looking the code, I see that
# 
#  - default lane color is "light yellow" (255, 217, 82). [code](https://github.com/lyft/l5kit/blob/1e235b8617488e818be30cd7193d43588125bbab/l5kit/l5kit/rasterization/semantic_rasterizer.py#L198)
#  - green, yellow, red color on lane is to show trafic light condition. [code](https://github.com/lyft/l5kit/blob/1e235b8617488e818be30cd7193d43588125bbab/l5kit/l5kit/rasterization/semantic_rasterizer.py#L199-L201)
#  - orange box represents crosswalk. [code](https://github.com/lyft/l5kit/blob/1e235b8617488e818be30cd7193d43588125bbab/l5kit/l5kit/rasterization/semantic_rasterizer.py#L204-L211)
# 
# Please refer below animation to verify it.

# In[ ]:


def create_animate_for_indexes(dataset, indexes):
    images = []
    timestamps = []

    for idx in indexes:
        data = dataset[idx]
        im = data["image"].transpose(1, 2, 0)
        im = dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
        center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
        clear_output(wait=True)
        images.append(PIL.Image.fromarray(im[::-1]))
        timestamps.append(data["timestamp"])

    anim = animate_solution(images, timestamps)
    return anim

def create_animate_for_scene(dataset, scene_idx):
    indexes = dataset.get_scene_indices(scene_idx)
    return create_animate_for_indexes(dataset, indexes)


# Car was stopping during traffic light is red, and starts once traffic becomes green.

# In[ ]:


dataset = dataset_dict["py_semantic"]
scene_idx = 34
anim = create_animate_for_scene(dataset, scene_idx)
print("scene_idx", scene_idx)
HTML(anim.to_jshtml())


# In[ ]:


scene_idx = 0
print("scene_idx", scene_idx)
anim = create_animate_for_scene(dataset, scene_idx)
display(HTML(anim.to_jshtml()))


# We can see car comes to the red traffic light and stopped.

# In[ ]:


scene_idx = 1
print("scene_idx", scene_idx)
anim = create_animate_for_scene(dataset, scene_idx)
display(HTML(anim.to_jshtml()))


# Traffic light is always red in the scene. But agent car which turns right moves, which is allowed in US.

# In[ ]:


scene_idx = 2
print("scene_idx", scene_idx)
anim = create_animate_for_scene(dataset, scene_idx)
display(HTML(anim.to_jshtml()))


# ## Observation
# 
# We understand that `SemanticRasterizer` creates each lane, traffic light status or car's box as **"RGB" image**.<br/>
# However it is not necessary for CNN model to input RGB image. Any information representation for each channel is allowed.
# 
# I guess different representation may boost the prediction model's performance, which you need to write your own rasterizer.<br/>
# For example represent own car, other car, lane and traffic light in different channel.

# <a id="ego_agent_dataset"></a>
# # 2. Understanding EgoDataset/AgentDataset class
# 
# Instead of working with raw data, L5Kit provides PyTorch ready datasets.
# It's much easier to use this wrapped dataset class to access data.
# 
# 2 dataset class is implemented.
# 
#  - **EgoDataset**: this dataset iterates over the AV (Autonomous Vehicle) annotations
#  - **AgentDataset**: this dataset iterates over other agents annotations
# 
# Let's see each class in detail. What kind of attributes/methods they have? What kind of data structure for each attributes?
# 
# 
# As written in [Class diagram](#class_diagram), both classes are instantiated by:
#  - `cfg`: configuration file
#  - `ChunkedDataset`: Internal data class which holds 4 raw data `scenes`, `frames`, `agents` and `tl_faces` (described later).
#  - `rasterizer`: Rasterizer converts raw data into image.

# ## EgoDataset
# 
# 
# The implementation code is in [ego.py](https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/dataset/ego.py).<br/>
# 
# ### Internal data structure
# <img src="https://storage.googleapis.com/kaggle-forum-message-attachments/990934/16784/l5kit_ego_dataset.png" width="600" />
# 
# `EgoDataset` consists of multiple scenes. Each scene is usually 25 sec consecutive events, consists of multiple frames.<br/>
# A frame reprsents specific time's snapshot. Snapshot is taken in 0.1 sec interval, so usually 1 scene is made of about 250 frames.
# 
# Blue box represents each scene, orange box represents each frame as well as each data index.
# 
# When we access dataset by index i, i-th frame is returned. Frames are concatenated by multiple scenes, we different i-th index points different scene.
# The point where the scene will change is represented by `cumulative_sizes`.

# In[ ]:


semantic_rasterizer = rasterizer_dict["py_semantic"]
dataset = dataset_dict["py_semantic"]


# In[ ]:


# It shows the split point of each scene.
print("cumulative_sizes", dataset.cumulative_sizes)

# How's the length of each scene?
print("Each scene's length", dataset.cumulative_sizes[1:] - dataset.cumulative_sizes[:-1])


# It seems 1 scene usually consists of 248, 249 frames.
# 
# Now Let's check each method:
# 
# ### getitem, get_frame
# 
# When we access data by index as `dataset[i]`, `__getitem__` is called and l5kit internally calls `get_frame`.<br/>
# This method preprocesses the data and returns many features as dict format.

# In[ ]:


data = dataset[0]

print("dataset[0]=data is ", type(data))

def _describe(value):
    if hasattr(value, "shape"):
        return f"{type(value).__name__:20} shape={value.shape}"
    else:
        return f"{type(value).__name__:20} value={value}"

for key, value in data.items():
    print("  ", f"{key:25}", _describe(value))


# Each attribute represents follows (The data structure is same for `AgentDataset`, and I included explanation for `AgentDataset` as well):
# 
#  - image: image drawn by Rasterizer. As you saw on the top of this kernel. This is usually be the **input image for CNN**
#  - target_positions: The "Ego car" or "Agent (car/cyclist/pedestrian etc)"'s future position. This is **the value to predict in this competition (not for Ego car's, but for Agents)**.
#  - target_yaws: The Ego car's future yaw, to represent heading direction.
#  - target_availabilities: flag to represent this is valid or not. Only flag=1 is used for competition evaluation.
#  - history_positions: Past positions
#  - history_yaws: Past yaws
#  - history_availabilities:
#  - world_to_image: 3x3 transformation matrix to convert world-coordinate into pixel-coordinate.
#  - track_id: Unique ID for each Agent. `None` for Ego car.
#  - timestamp: timestamp for current frame.
#  - centroid: current center position
#  - yaw: current direction
#  - extent: Ego car or Agent's size. The car is not represented as point, but should be cared as dot box to include size information on the map.

# If you don't know what is "yaw", please refer [wikipedia](https://en.wikipedia.org/wiki/Yaw_(rotation)).
# 
# <div style="clear:both;display:table">
# <img src="https://upload.wikimedia.org/wikipedia/commons/5/54/Flight_dynamics_with_text.png" style="width:30%;float:left"/>
# <img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Aileron_yaw.gif" style="width:30%;float:left"/>
# </div>

# Next method to check is **`get_scene_indices` and `get_scene_dataset`**.
# 
# - `get_scene_indices(i)` method will return i-th scene's frame indices.
# - `get_scene_dataset(i)` method will return other `EgoDataset` which only contains i-th scene's frames.
#    - As you can see below, it contains only 1 scene when we visualize whole dataset.

# In[ ]:


scene_index = 0
frame_indices = dataset.get_scene_indices(scene_index)
print(f"frame_indices for scene {scene_index} = {frame_indices}")

scene_dataset = dataset.get_scene_dataset(scene_index)
print(f"scene_dataset {type(scene_dataset).__name__}, length {len(scene_dataset)}")

# Animate whole "scene_dataset"
create_animate_for_indexes(scene_dataset, np.arange(len(scene_dataset)))


# Last method I will explain is `get_frame_indices`.
# 
# `get_frame_indices(j)` will return all the `dataset[i]` whose frame points to `j`-th frame.<br/>
# For `EgoDataset`, these are same and i-th data always points to i-th frame.

# In[ ]:


frame_idx = 10
indices = dataset.get_frame_indices(frame_idx)

# These are same for EgoDataset!
print(f"frame_idx = {frame_idx}, indices = {indices}")


# ## AgentDataset

# 
# 
# The implementation code is in [agent.py](https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/dataset/agent.py).<br/>
# 
# ### Internal data structure
# <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F518134%2F9ea7ae3af0037edb95cc0effe361568f%2Fl5kit_agent_dataset.png?generation=1598749570093014&alt=media" width="600" />
# 
# `AgentDataset` consists of same multiple scenes as `EgoDataset`.
# Always 1 host Ego car exists in each frame, however the number of agent differ.<br/>
# It can contain multiple agents in 1 frame, or even 0 agents in some of the frames.
# 
# See above figure, AgentDataset contains all the agents' frame in different index.<br/>
# If you access `dataset[i]`, it returns unique scene's unique frame's unique agent's information.
# 
# In the figure, the x-axis represents same frame = same timestamp, and the y-axis represents agents.
# Blue box represents each scene, black box represents same frame, and orange box represents each data index.
# 
# The point where the scene will change is represented by `cumulative_sizes`, same as `EgoDataset`.
# The point where the **agent** will change is represented by **`cumulative_sizes_agents`**.

# In[ ]:


semantic_rasterizer = rasterizer_dict["py_semantic"]
agent_dataset = AgentDataset(cfg, zarr_dataset, semantic_rasterizer)

print(f"EgoDataset size {len(dataset)}, AgentDataset size {len(agent_dataset)}")


# `AgentDataset` size is usually much bigger than `EgoDataset`, since multiple agents exist in 1 frame.
# 
# The returned data structure by `__getitem__` is same with `EgoDataset`. Please refer `EgoDataset` exlanation for details of each key.

# In[ ]:


# The returned data structure is same.
data = agent_dataset[0]

print("agent_dataset[0]=data is ", type(data))

def _describe(value):
    if hasattr(value, "shape"):
        return f"{type(value).__name__:20} shape={value.shape}"
    else:
        return f"{type(value).__name__:20} value={value}"

for key, value in data.items():
    print("  ", f"{key:25}", _describe(value))


# Same methods with `EgoDataset` are supported:
# 
# - `get_scene_indices(i)` method will return i-th scene's frame indices. However it contains multiple agents and **thus contains same frame multiple times**.
# - `get_scene_dataset(i)` method will return other `AgentDataset` which only contains i-th scene's frames.
# 
# Please see below animation.<br/>
# Same scene contains multiple agents, thus contains multiple same frames. Recommended to press "Next frame" button manually to how timestamp=frame evolves.

# In[ ]:


scene_index = 3
frame_indices = agent_dataset.get_scene_indices(scene_index)
print(f"frame_indices for scene {scene_index} = {frame_indices}")

scene_dataset = agent_dataset.get_scene_dataset(scene_index)
print(f"scene_dataset {type(scene_dataset).__name__}, length {len(scene_dataset)}")

# Animate whole "scene_dataset"
create_animate_for_indexes(scene_dataset, np.arange(len(scene_dataset)))


# `get_frame_indices` returns the index which contains same frame.
# 
# `EgoDataset` return was trivial, but `AgentDataset` may return multiple indices since several agents exist in each frame. 
# Let's see example:

# In[ ]:


for i in range(1000):
    print(i, agent_dataset.get_frame_indices(i))


# In[ ]:


frame_indices = agent_dataset.get_frame_indices(648)

fig, axes = plt.subplots(1, len(frame_indices), figsize=(15, 5))
axes = axes.flatten()

for i in range(len(frame_indices)):
    index = frame_indices[i]
    t = agent_dataset[index]["timestamp"]
    # Timestamp is same for same frame.
    print(f"timestamp = {t}")
    visualize_rgb_image(agent_dataset, index=index, title=f"index={index}", ax=axes[i])
fig.show()


# <a id="raw_data"></a>
# # 3. Understanding raw data structures
# 
# We have seen `EgoDataset` & `AgentDataset` functionality, however it is a wrapped methods and we have still not understanding how the raw data is made.<br/>
# Let's focus on `ChunkDataset` (`zarr_dataset` variable) here, it contains **scenes, frames, agents, tl_faces** attributes, which are the raw structured array data.
# 
# Each data structure definition can be checked at [here](https://github.com/lyft/l5kit/blob/master/data_format.md#2020-lyft-competition-dataset-format).
# 
# ### Overfiew figure
# 
# <img src="https://storage.googleapis.com/kaggle-forum-message-attachments/991027/16786/l5kit_chunked_dataset.png" width="600" />
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
# `scenes` data contains each scene's information.<br/>
# Each scene basically owns a reference to the frames and consists of following information:
# 
#  - frame_index_interval: frame index for this scene.
#  - host: unique name of host car.
#  - start_time, end_time: timestamp for start & end of scene.

# In[ ]:


print("scenes", zarr_dataset.scenes)
print("scenes[0]", zarr_dataset.scenes[0])


# Now we understand that scene 0 consists of frame 0~248. Let's see go to see frame.
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
# `frames` data contains each frame's information.<br/>
# Each frame owns a reference to the agents & traffic_light_faces.<br/>
# It consists of following information:
# 
#  - timestamp: timestamp of this frame.
#  - agent_index_interval: agent index for this frame.
#  - traffic_light_faces_index_interval: traffic light faces index for this frame.
#  - ego_translation, ego_rotation: position & direction of the host Ego car.
# 
# frame does not contain any "image" information. image is created by Rasterizer by just the world coordinate position information known by `ego_translation` using `MapAPI`.

# In[ ]:


print("frames", zarr_dataset.frames)
print("frames[0]", zarr_dataset.frames[0])


# We see that frame 0 contains
#  - agents for [0, 38).
#  - traffic lights for [0, 0) (No traffic light included in this frame).

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
# `agents` data contains **each frame's agent (vehicles, cyclists and pedestrians)** information.<br/>
# It consists of following information:
# 
#  - centroid: position of the agent.
#  - extent: the size of the agent.
#  - yaw: direction of the agent.
#  - velocity: current velocity of the agent.
#  - track_id: unique id to represent same agent within **different frames**.
#  - label_probabilities: The agent is identification is automated by using already trained percenption network. Thus what kind of type is provided by predicted proability.
# 
# The label definition can be found [here](https://github.com/lyft/l5kit/blob/1e235b8617488e818be30cd7193d43588125bbab/l5kit/l5kit/data/labels.py#L1-L19).
# We can understand the agent is either a car, a cyclist, a pedestrian etc.

# In[ ]:


print("agents", zarr_dataset.agents)
print("agents[0]", zarr_dataset.agents[0])


# ### traffic_light_faces
# 
# ```
# TL_FACE_DTYPE = [
#     ("face_id", "<U16"),
#     ("traffic_light_id", "<U16"),
#     ("traffic_light_face_status", np.float32, (len(TL_FACE_LABELS,))),
# ]
# ```
# 
# `tl_faces` data contains **each frame's traffic light** information.<br/>
# It consists of following information:
# 
#  - face_id: unique id for the traffic light bulb. Note that unlike agent, this traffic light is unique on the map for all scenes.
#  - traffic_light_id: represent a traffic light status, e.g. the light is green/yellow/red etc. See [protocol buffer definition](https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/data/proto/road_network.proto#L615) for details.
#  - traffic_light_status: 3-dim array. I guess each represents the condition of green/yellow/red light. Condition definition is [here](https://github.com/lyft/l5kit/blob/1e235b8617488e818be30cd7193d43588125bbab/l5kit/l5kit/data/labels.py#L23-L27), i.e., Active/Inactive/Unknown.
# 

# In[ ]:


print("tl_faces", zarr_dataset.tl_faces)
print("tl_faces[0]", zarr_dataset.tl_faces[0])


# # Next to go
# 
# That's all for going deep into the l5kit library.
# 
# We can write your own rasterizer or even manually handling raw data to create more informative input/output data for prediction model to achieve high accuracy.
# 
# **[Update] I wrote a kernel to train prediction models for competition submission as next topic to try!**
#  - **[Lyft: Training with multi-mode confidence](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence)**
#  - [Lyft: Prediction with multi-mode confidence](https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence)

# # Further reference
# 
#  - Paper of this Lyft Level 5 prediction dataset: [One Thousand and One Hours: Self-driving Motion Prediction Dataset](https://arxiv.org/abs/2006.14480)
#  - [jpbremer/lyft-scene-visualisations](https://www.kaggle.com/jpbremer/lyft-scene-visualisations)

# <h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>
