#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget https://www.tbi.univie.ac.at/RNA/download/ubuntu/ubuntu_18_04/viennarna_2.4.15-1_amd64.deb')
get_ipython().system('apt-get install ./viennarna_2.4.15-1_amd64.deb -y')
get_ipython().system('git clone https://github.com/DasLab/arnie')

get_ipython().system('/opt/conda/bin/python3.7 -m pip install --upgrade pip')
get_ipython().system('git clone https://www.github.com/DasLab/draw_rna draw_rna_pkg')
get_ipython().system('cd draw_rna_pkg && python setup.py install')

get_ipython().system("yes '' | cpan -i Graph")
get_ipython().system('git clone https://github.com/hendrixlab/bpRNA')


# ## Setting

# In[ ]:


import os
import sys

get_ipython().system('echo "vienna_2: /usr/bin" > arnie.conf')
get_ipython().system('echo "TMP: /kaggle/working/tmp" >> arnie.conf')
get_ipython().system('mkdir -p /kaggle/working/tmp')
os.environ["ARNIEFILE"] = f"/kaggle/working/arnie.conf"
sys.path.append('/kaggle/working/draw_rna_pkg/')
sys.path.append('/kaggle/working/draw_rna_pkg/ipynb/')
pkg = 'vienna_2'


# In[ ]:



import numpy as np
import pandas as pd
from multiprocessing import Pool
from arnie.pfunc import pfunc
from arnie.mea.mea import MEA
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
from tqdm.notebook import tqdm as tqdm

n_candidates = 2
# turn off for all data
debug = True


# In[ ]:


get_ipython().system('grep processor /proc/cpuinfo | wc -l')


# In[ ]:


MAX_THRE = 4


# In[ ]:


train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
if debug:
    train = train[:20]
    test = test[:20]
target_df = train.append(test)


# ## Getting structure
# 

# In[ ]:


def proc1(arg):
    sequence = arg[0]
    id = arg[1]
    log_gamma = arg[2]
    bp_matrix = bpps(sequence, package=pkg)
    mea_mdl = MEA(bp_matrix,gamma=10**log_gamma)
    return id, sequence, mea_mdl.structure, log_gamma, mea_mdl.score_expected()[2]

li = []
for log_gamma in range(10):
    for i, arr in enumerate(target_df[['sequence','id']].values):
        li.append([arr[0], arr[1], log_gamma])

p = Pool(processes=MAX_THRE)
results = []
for ret in tqdm(p.imap(proc1, li),total=len(li)):
    results.append(ret)
    #print(f'done for {ret[0]}')
df = pd.DataFrame(results, columns=['id', 'sequence', 'structure', 'log_gamma', 'score'])

df_tmp = target_df[['id', 'sequence', 'structure']].copy()
df_tmp['log_gamma'] = 100
df_tmp['score'] = 100
df = df.append(df_tmp).sort_values('score', ascending=False).reset_index(drop=True)

new_df = pd.DataFrame()
for id in df['id'].unique():
    unq_df = df[df['id'] == id].drop_duplicates('structure')
    unq_df['cnt'] = unq_df.shape[0]
    new_df = new_df.append(unq_df[1:min(n_candidates,len(unq_df))])


# ## Getting predicted_loop_type
# 

# In[ ]:


get_ipython().system('mkdir -p tmp_files')
def get_predicted_loop_type(id, sequence, structure, debug=False):
    structure_fixed = structure.replace('.','0').replace('(','1').replace(')','2')
    pid = os.getpid()
    tmp_in_file = f'tmp_files/{id}_{structure_fixed}_{pid}.dbn'
    tmp_out_file = f'{id}_{structure_fixed}_{pid}.st'
    get_ipython().system('echo $sequence > $tmp_in_file')
    get_ipython().system('echo "$structure" >> $tmp_in_file')
    get_ipython().system('export PERL5LIB=/root/perl5/lib/perl5 && perl bpRNA/bpRNA.pl $tmp_in_file')
    result = [l.strip('\n') for l in open(tmp_out_file)]
    if debug:
        print(sequence)
        print(structure)
        print(result[5])
    else:
        get_ipython().system('rm $tmp_out_file $tmp_in_file')
    return id, structure, result[5]

def proc2(arg):
    result = get_predicted_loop_type(arg[0], arg[1], arg[2], debug=False)
    return result

li = []
for i, arr in enumerate(new_df[['id', 'sequence', 'structure']].values):
    li.append(arr)

p = Pool(processes=MAX_THRE)
results_loop_type = []
for ret in tqdm(p.imap(proc2, li),total=len(li)):
    results_loop_type.append(ret)
    #print(f'done for {ret[0]}')

new_df = new_df.merge(pd.DataFrame(results_loop_type, columns=('id', 'structure', 'predicted_loop_type')), on=['id','structure'], how='left')
new_df.to_csv('aug_data.csv', index=False)


# In[ ]:


new_df.head()


# In[ ]:




