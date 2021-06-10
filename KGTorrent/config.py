"""
This is the configuration file of KGTorrent.

Here the main variables of the program are set, mostly by reading their values from environment variables.

See :ref:`configuration` for details on the environment variables that must be set to run KGTorrent.
"""

import logging
import os
import time

# MySQL DB configuration
db_host = 'localhost'
db_port = '3308'
db_name = 'kgtorrent'
db_username = 'jyang'
db_password = 'nqpdjY3Y'

# Data paths
#meta_kaggle_path = os.environ['METAKAGGLE_PATH']
meta_kaggle_path = "./tmp/"
constraints_file_path = '../data/fk_constraints_data.csv'

# Notebook dataset configuration
#nb_archive_path = os.environ['NB_DEST_PATH']
nb_archive_path = 'C:/Users/jimmy/Documents/GitHub/KGTorrent/notebooks'

nb_conf = {
    'languages': ['IPython Notebook HTML']
}

# Logging Configuration
#log_path = os.environ['LOG_DEST_PATH']
log_path = "./logs/"
# logging.basicConfig(
#     filename=os.path.join(log_path, f'{time.time()}.log'),
#     filemode='w',
#     level=logging.INFO,
#     format='[%(levelname)s]\t%(asctime)s - %(message)s'
# )
