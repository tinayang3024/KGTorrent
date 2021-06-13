#!/usr/bin/env python
# coding: utf-8

# # Installation
# The first step is to install required tools as well:

# In[ ]:


# Install:
# Kaggle environments.
get_ipython().system('git clone -q https://github.com/Kaggle/kaggle-environments.git')
get_ipython().system('cd kaggle-environments && pip install -q .')
# GFootball environment.
get_ipython().system('apt-get update -qy ')
get_ipython().system('apt-get install -qy libsdl2-gfx-dev libsdl2-ttf-dev')
# Make sure that the Branch in git clone and in wget call matches !!
get_ipython().system('git clone -b v2.3 https://github.com/google-research/football.git')
get_ipython().system('mkdir -p football/third_party/gfootball_engine/lib')
get_ipython().system('wget -q --show-progress https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.3.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so')
get_ipython().system('cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install -q .')


# # GFootball Rule-based Bot Template and Environment Exploration
# Below we present a simple Bot with couple rules playing GFootball. 
# In this one, I'm trying to use the log file to further understand the environment, and try my best to summarize some rules based on my game experience.
# I would like to call rule-based agent as GFootball coach simulator.
# It is not simulating how you coaching a football team. 
# Instead, it is simulating how you shout to your roommate, "Why you idiot do not press RL+A+X when you blublublu?", when your roommate is playing and you are watching. :)
# What is better now, your roommate can achieve some operations human-being cannot achieve! :)
# 
# Do go through the detailed description of the GFootball observations is available [here](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md).
# Also the helper code to see the enum structure at [here](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/football/helpers.py).
# They are very helpful.
# And then, we can start coding the rules. There are couple high-level tricks which can be summarized from playing the game as well:
# 1. **Try to check sticky_actions before you do slide, pass, and shot, epecially the direction part.** It is simulating your direction button holding action when you use controller to paly the game.
# 2. The ball ground level height is around 0.10 - 0.15 and player can pick a high pass at height around 0.5-1.0, while goalkeeper can catch the ball at around 0.5-2.0. Gravity is around 0.098, and drag plays a role during ball flying.
# 3. The sprint speed is around 0.015 per step
# 4. When PalyerRole is involved, there will be errors somehow, which is blocking to locate enemy goalkeeper accurately. My assumption here is that the latest PR merged in the helper is not available in the scoring environment. Will try to use it later.
# 4. The space between boal controller and ball is around 0.012 during sprint which is close to the sprint speed. We may assume this is equal to one-step running length.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', "import numpy as np\nfrom kaggle_environments.envs.football.helpers import *\n\n@human_readable_agent\ndef agent(obs):\n    \n    # Global param\n    goal_threshold = 0.5\n    gravity = 0.098\n    pick_height = 0.5\n    step_length = 0.015 # As we always sprint\n    body_radius = 0.012\n    slide_threshold = step_length + body_radius\n    \n    # Ignore drag to estimate the landing point\n    def ball_landing(ball, ball_direction):\n        start_height = ball[2]\n        end_height = pick_height\n        start_speed = ball_direction[2]\n        time = np.sqrt(start_speed**2/gravity**2 - 2/gravity*(end_height-start_height)) + start_speed/gravity\n        return [ball[0]+ball_direction[0]*time, ball[1]+ball_direction[1]*time]\n    \n    # Check whether pressing on direction buttons and take action if so\n    # Else press on direction first\n    def sticky_check(action, direction):\n        if direction in obs['sticky_actions']:\n            return action\n        else:\n            return direction\n    \n    # Find right team positions\n    def_team_pos = obs['right_team']\n    # Fix goalkeeper index here as PlayerRole has issues\n    # Default PlayerRole [0, 7, 9, 2, 1, 1, 3, 5, 5, 5, 6]\n    def_keeper_pos = obs['right_team'][0]\n    \n    # We always control left team (observations and actions\n    # are mirrored appropriately by the environment).\n    controlled_player_pos = obs['left_team'][obs['active']]\n    # Get team size\n    N = len(obs['left_team'])\n    \n    # Does the player we control have the ball?\n    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:\n        # Kickoff strategy: short pass to teammate\n        if obs['game_mode'] == GameMode.KickOff:\n            return sticky_check(Action.ShortPass, Action.Top) if controlled_player_pos[1] > 0 else sticky_check(Action.ShortPass, Action.Bottom)\n        # Goalkick strategy: high pass to front\n        if obs['game_mode'] == GameMode.GoalKick:\n            return sticky_check(Action.LongPass, Action.Right)\n        # Freekick strategy: make shot when close to goal, high pass when in back field, and short pass in mid field\n        if obs['game_mode'] == GameMode.FreeKick:\n            if controlled_player_pos[0] > goal_threshold:\n                if abs(controlled_player_pos[1]) < 0.1:\n                    return sticky_check(Action.Shot, Action.Right)\n                if abs(controlled_player_pos[1]) < 0.3:\n                    return sticky_check(Action.Shot, Action.TopRight) if controlled_player_pos[1]>0 else sticky_check(Action.Shot, Action.BottomRight)\n                return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)\n            \n            if controlled_player_pos[0] < -goal_threshold:\n                if abs(controlled_player_pos[1]) < 0.3:\n                    return sticky_check(Action.HighPass, Action.Right)\n                return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)\n            \n            if abs(controlled_player_pos[1]) < 0.3:\n                return sticky_check(Action.ShortPass, Action.Right)\n            return sticky_check(Action.ShortPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.ShortPass, Action.Bottom)\n        # Corner strategy: high pass to goal area\n        if obs['game_mode'] == GameMode.Corner:\n            return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)\n        # Throwin strategy: short pass into field\n        if obs['game_mode'] == GameMode.ThrowIn:\n            return sticky_check(Action.ShortPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.ShortPass, Action.Bottom)\n        # Penalty strategy: make a shot\n        if obs['game_mode'] == GameMode.Penalty:\n            right_actions = [Action.TopRight, Action.BottomRight, Action.Right]\n            for action in right_actions:\n                if action in obs['sticky_actions']:\n                    return Action.Shot\n            return np.random.choice(right_actions)\n            \n        # Defending strategy\n        if controlled_player_pos[0] < -goal_threshold:\n            if abs(controlled_player_pos[1]) < 0.3:\n                return sticky_check(Action.HighPass, Action.Right)\n            return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)\n            \n        # Make sure player is running.\n        if Action.Sprint not in obs['sticky_actions']:\n            return Action.Sprint\n        \n        # Shot if we are 'close' to the goal (based on 'x' coordinate).\n        if controlled_player_pos[0] > goal_threshold:\n            if abs(controlled_player_pos[1]) < 0.1:\n                return sticky_check(Action.Shot, Action.Right)\n            if abs(controlled_player_pos[1]) < 0.3:\n                return sticky_check(Action.Shot, Action.TopRight) if controlled_player_pos[1]>0 else sticky_check(Action.Shot, Action.BottomRight)\n            elif controlled_player_pos[0] < 0.85:\n                return Action.Right\n            else:\n                return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)\n        \n        # Run towards the goal otherwise.\n        return Action.Right\n    else:\n        # when the ball is generally on the ground not flying\n        if obs['ball'][2] <= pick_height:\n            # Run towards the ball's left position.\n            if obs['ball'][0] > controlled_player_pos[0] + slide_threshold:\n                if obs['ball'][1] > controlled_player_pos[1] + slide_threshold:\n                    return Action.BottomRight\n                elif obs['ball'][1] < controlled_player_pos[1] - slide_threshold:\n                    return Action.TopRight\n                else:\n                    return Action.Right\n            elif obs['ball'][0] < controlled_player_pos[0] + slide_threshold:\n                if obs['ball'][1] > controlled_player_pos[1] + slide_threshold:\n                    return Action.BottomLeft\n                elif obs['ball'][1] < controlled_player_pos[1] - slide_threshold:\n                    return Action.TopLeft\n                else:\n                    return Action.Left\n            # When close to the ball, try to take over.\n            else:\n                return Action.Slide\n        # when the ball is flying\n        else:\n            landing_point = ball_landing(obs['ball'], obs['ball_direction'])\n            # Run towards the landing point's left position.\n            if landing_point[0] - body_radius > controlled_player_pos[0] + slide_threshold:\n                if landing_point[1] > controlled_player_pos[1] + slide_threshold:\n                    return Action.BottomRight\n                elif landing_point[1] < controlled_player_pos[1] - slide_threshold:\n                    return Action.TopRight\n                else:\n                    return Action.Right\n            elif landing_point[0] - body_radius < controlled_player_pos[0] + slide_threshold:\n                if landing_point[1] > controlled_player_pos[1] + slide_threshold:\n                    return Action.BottomLeft\n                elif landing_point[1] < controlled_player_pos[1] - slide_threshold:\n                    return Action.TopLeft\n                else:\n                    return Action.Left\n            # Try to take over the ball if close to the ball.\n            elif controlled_player_pos[0] > goal_threshold:\n                # Keep making shot when around landing point\n                return sticky_check(Action.Shot, Action.Right) if ['ball'][2] <= pick_height else Action.Idle\n            else:\n                return sticky_check(Action.Slide, Action.Right) if ['ball'][2] <= pick_height else Action.Idle")


# In[ ]:


# Set up the Environment.
from kaggle_environments import make
env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})
output = env.run(["/kaggle/working/submission.py", "do_nothing"])[-1]
print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))
print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))
env.render(mode="human", width=800, height=600)


# In[ ]:


# Validation
from datetime import datetime
from kaggle_environments import make
start = datetime.now()
env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})
output = env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])[-1]
print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))
print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))
print(datetime.now()-start)
env.render(mode="human", width=800, height=600)


# Load out log files from environment.

# In[ ]:


import pandas as pd
log = pd.DataFrame(env.steps)


# Take a look at left team log for beginning steps.

# In[ ]:


log[0].head()


# A further look into one step.

# In[ ]:


log.iloc[0,0]


# What I want to explore is the gravity constant in this game. So I keep only ball position and speed log only.

# In[ ]:


ball_log = pd.DataFrame()
ball_log['ball'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball'])
ball_log['ball_direction'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball_direction'])
ball_log.head(20)


# It seems the ball will stay at center for couple steps before the game start. Also we can explore how the speed stored in ball_direction work.

# In[ ]:


print('Ball position at step 9 is ', ball_log.iloc[9,0])
print('Ball position at step 10 is ', ball_log.iloc[10,0])
print('Ball speed at step 9 is ', ball_log.iloc[9,1])
print('Ball speed at step 10 is ', ball_log.iloc[10,1])
print('Ball position change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[9,0], ball_log.iloc[10,0])])
print('Ball speed change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[9,1], ball_log.iloc[10,1])])


# In[ ]:


print('Ball position at step 9 is ', ball_log.iloc[8,0])
print('Ball position at step 10 is ', ball_log.iloc[9,0])
print('Ball speed at step 9 is ', ball_log.iloc[8,1])
print('Ball speed at step 10 is ', ball_log.iloc[9,1])
print('Ball position change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[8,0], ball_log.iloc[9,0])])
print('Ball speed change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[8,1], ball_log.iloc[9,1])])


# Above results will chagne game to game, what I saw in a game:
# 
# Ball position at step 9 is  [0.03627660498023033, 0.0030293932650238276, 0.4980185627937317]
# 
# Ball position at step 10 is  [0.04673447832465172, 0.003420717315748334, 0.39738738536834717]
# 
# Ball speed at step 9 is  [0.010506737977266312, 0.00039332741289399564, -0.047481197863817215]
# 
# Ball speed at step 10 is  [0.010418068617582321, 0.00038968087756074965, -0.14398618042469025]
# 
# Ball position change between step 9 and 10 is  [0.010457873344421387, 0.0003913240507245064, -0.10063117742538452]
# 
# Ball speed change between step 9 and 10 is  [-8.866935968399048e-05, -3.6465353332459927e-06, -0.09650498256087303]
# 
# 
# Ball position at step 9 is  [0.025729181244969368, 0.002634421456605196, 0.5017596483230591]
# 
# Ball position at step 10 is  [0.03627660498023033, 0.0030293932650238276, 0.4980185627937317]
# 
# Ball speed at step 9 is  [0.01059782411903143, 0.00039699606713838875, 0.04987070709466934]
# 
# Ball speed at step 10 is  [0.010506737977266312, 0.00039332741289399564, -0.047481197863817215]
# 
# Ball position change between step 9 and 10 is  [0.010547423735260963, 0.00039497180841863155, -0.0037410855293273926]
# 
# Ball speed change between step 9 and 10 is  [-9.108614176511765e-05, -3.6686542443931103e-06, -0.09735190495848656]
# 
# We can easily see the air fraction can be ignored generally, and the gravity constant in the game is around 0.097.
# 

# And below we are going to explore what is the speed during sprint.

# In[ ]:


right1 = pd.DataFrame()
right1['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['right_team'][1])
right1['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['right_team_direction'][1])


# In[ ]:


print('Right team player 1 position at step 35 is ',right1['position'][35])
print('Right team player 1 position at step 36 is ',right1['position'][36])
print('Right team player 1 speed at step 35 is ',right1['speed'][35])
print('Right team player 1 speed at step 36 is ',right1['speed'][36])
print('Right team player 1 position change at step 35 is ',[b - a for a, b in zip(right1.iloc[35,0], right1.iloc[36,0])])
print('Right team player 1 speed change at step 35 is ',[b - a for a, b in zip(right1.iloc[35,1], right1.iloc[36,1])])


# Per the video of the game I saw, right team player 1 was sprinting with the ball during steps 30 to 40. So I specifically checked the palyer log during those steps.
# 
# Right team player 1 position at step 35 is  [-0.14991192519664764, 0.028422148898243904]
# 
# Right team player 1 position at step 36 is  [-0.16384349763393402, 0.028232717886567116]
# 
# Right team player 1 speed at step 35 is  [-0.013850015588104725, -0.0001925030373968184]
# 
# Right team player 1 speed at step 36 is  [-0.013992365449666977, -0.00017117084644269198]
# 
# Right team player 1 position change at step 35 is  [-0.013931572437286377, -0.00018943101167678833]
# 
# Right team player 1 speed change at step 35 is  [-0.00014234986156225204, 2.1332190954126418e-05]
# We can see that the sprint speed is around 0.014 per step with ball

# In[ ]:


step = 70
player = pd.DataFrame()
player['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team'][8])
player['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team_direction'][8])
ball = pd.DataFrame()
ball['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball'])
ball['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball_direction'])
print('Player position at step ',step,' is ',player['position'][step])
print('Ball position at step ',step,' is ',ball['position'][step])
print('Player position at step ',step+1,' is ',player['position'][step+1])
print('Ball position at step ',step+1,' is ',ball['position'][step+1])
print('Player position at step ',step+2,' is ',player['position'][step+2])
print('Ball position at step ',step+2,' is ',ball['position'][step+2])
print('Player position at step ',step+3,' is ',player['position'][step+3])
print('Ball position at step ',step+3,' is ',ball['position'][step+3])


# In the game I saw, during step 60-80, player 8 is sprinting toward right with ball. 
# We can see there is always a 0.01-0.015 space between player and ball.
# This space is very close to the sprint length per step.
# 
# Player position at step  70  is  [0.4324444532394409, 0.015978505834937096]
# 
# Ball position at step  70  is  [0.4440920054912567, 0.01583743654191494, 0.11439678072929382]
# 
# Player position at step  71  is  [0.446736216545105, 0.01591404154896736]
# 
# Ball position at step  71  is  [0.4558497369289398, 0.01592208817601204, 0.1119992733001709]
# 
# Player position at step  72  is  [0.4609338641166687, 0.015880394726991653]
# 
# Ball position at step  72  is  [0.46899694204330444, 0.015966957435011864, 0.14346031844615936]
# 
# Player position at step  73  is  [0.4747302532196045, 0.01554815098643303]
# 
# Ball position at step  73  is  [0.48480528593063354, 0.015958167612552643, 0.15489338338375092]
# 

# In[ ]:


step = 150
player = pd.DataFrame()
player['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team'][9])
player['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team_direction'][9])
ball = pd.DataFrame()
ball['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball'])
ball['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball_direction'])
for i in range(5):
    print('Player position at step ',step+i,' is ',player['position'][step+i])
    print('Ball position at step ',step+i,' is ',ball['position'][step+i])
    print('Player speed at step ',step+i,' is ',player['speed'][step+i])
    print('Ball speed at step ',step+i,' is ',ball['speed'][step+i])


# In[ ]:




