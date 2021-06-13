#!/usr/bin/env python
# coding: utf-8

# # GFootball Template Bot
# Below we present a simple Bot playing GFootball. The first step is to install required tools:

# In[1]:


# Install:
# Kaggle environments.
get_ipython().system('git clone https://github.com/Kaggle/kaggle-environments.git')
get_ipython().system('cd kaggle-environments && pip install .')

# GFootball environment.
get_ipython().system('apt-get update -y')
get_ipython().system('apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev')

# Make sure that the Branch in git clone and in wget call matches !!
get_ipython().system('git clone -b v2.8 https://github.com/google-research/football.git')
get_ipython().system('mkdir -p football/third_party/gfootball_engine/lib')

get_ipython().system('wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so')
get_ipython().system('cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .')


# Now it is time to implement a bot. It consists of a single function **agent**, which is called by the Kaggle Environment on each step.
# **Agent** receives Kaggle-specific observations as a parameter, which contain GFootball environment observations under **players_raw** key.
# Detailed description of the GFootball observations is available [here](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md).

# In[2]:


get_ipython().run_cell_magic('writefile', 'submission.py', "from kaggle_environments.envs.football.helpers import *\n\n# @human_readable_agent wrapper modifies raw observations \n# provided by the environment:\n# https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations\n# into a form easier to work with by humans.\n# Following modifications are applied:\n# - Action, PlayerRole and GameMode enums are introduced.\n# - 'sticky_actions' are turned into a set of active actions (Action enum)\n#    see usage example below.\n# - 'game_mode' is turned into GameMode enum.\n# - 'designated' field is removed, as it always equals to 'active'\n#    when a single player is controlled on the team.\n# - 'left_team_roles'/'right_team_roles' are turned into PlayerRole enums.\n# - Action enum is to be returned by the agent function.\n@human_readable_agent\ndef agent(obs):\n    # Make sure player is running.\n    if Action.Sprint not in obs['sticky_actions']:\n        return Action.Sprint\n    # We always control left team (observations and actions\n    # are mirrored appropriately by the environment).\n    controlled_player_pos = obs['left_team'][obs['active']]\n    # Does the player we control have the ball?\n    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:\n        # Shot if we are 'close' to the goal (based on 'x' coordinate).\n        if controlled_player_pos[0] > 0.5:\n            return Action.Shot\n        # Run towards the goal otherwise.\n        return Action.Right\n    else:\n        # Run towards the ball.\n        if obs['ball'][0] > controlled_player_pos[0] + 0.05:\n            return Action.Right\n        if obs['ball'][0] < controlled_player_pos[0] - 0.05:\n            return Action.Left\n        if obs['ball'][1] > controlled_player_pos[1] + 0.05:\n            return Action.Bottom\n        if obs['ball'][1] < controlled_player_pos[1] - 0.05:\n            return Action.Top\n        # Try to take over the ball if close to the ball.\n        return Action.Slide")


# In[3]:


# Set up the Environment.
from kaggle_environments import make
env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})
output = env.run(["/kaggle/working/submission.py", "do_nothing"])[-1]
print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))
print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))
env.render(mode="human", width=800, height=600)


# # Submit to Competition
# 1. "Save & Run All" (commit) this Notebook
# 1. Go to the notebook viewer
# 1. Go to "Data" section and find submission.py file.
# 1. Click "Submit to Competition"
# 1. Go to [My Submissions](https://www.kaggle.com/c/football/submissions) to view your score and episodes being played.
