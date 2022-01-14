"""
This module defines the class that handles the actual download of Jupyter notebooks from Kaggle.
"""

import logging
import time
from pathlib import Path
from tqdm import tqdm

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

# Imports for testing
import KGTorrent.config as config
from KGTorrent.db_communication_handler import DbCommunicationHandler

import glob
import os
import re

class Downloader:
    """
    The ``Downloader`` class handles the download of Jupyter notebooks from Kaggle.
    It needs the notebook slugs and identifiers ``pandas.DataFrame`` in order to request
    notebooks from Kaggle.
    To do so it uses one of the following two strategies:

    ``HTTP``
        to download full notebooks via HTTP requests;

    ``API``
        to download notebooks via calls to the official Kaggle API;
        Jupyter notebooks downloaded by using this strategy always miss the output of code cells.

    Notebooks that are already present in the download folder are skipped.
    During the ``refresh`` procedure all those notebooks that are already present in the download folder
    but are no longer referenced in the KGTorrent database are deleted.
    """

    def __init__(self, nb_identifiers, nb_archive_path):
        """
        The constructor of this class sets notebook identifiers and download folder provided by the arguments.
        It also initializes the counters for successes and failures.

        Args:
            nb_identifiers: The ``pandas.DataFrame`` containing notebook slugs and identifiers.
            nb_archive_path: The path to the download folder.
        """

        # Notebook slugs and identifiers [UserName, CurrentUrlSlug, CurrentKernelVersionId]
        self._nb_identifiers = nb_identifiers

        # Destination Folder
        self._nb_archive_path = nb_archive_path

        # Counters for successes and failures
        self._n_successful_downloads = 0
        self._n_failed_downloads = 0

    def _check_destination_folder(self):
        """
        This method verifies the bond between notebooks in the download folder and the identifiers
        in the notebook slugs and identifiers ``pandas.DataFrame``.
        It checks whether an identifier of the notebooks, which are present in the destination folder,
        is present in the notebook slugs and identifiers ``pandas.DataFrame``.
        If present it deletes the notebook identifiers from the notebook slugs and identifiers ``pandas.DataFrame``.
        If not present it deletes the bondless notebook in the download folder.
        """

        # Get notebook names
        notebook_paths = list(Path(self._nb_archive_path).glob('*.ipynb'))

        for path in notebook_paths:
            name = path.stem
            split = name.split('_')

            # check if the file have valid name
            if len(split) == 2:

                # If the file exists in folder drop it from res
                if (split[0] in self._nb_identifiers['UserName'].values) & \
                        (split[1] in self._nb_identifiers['CurrentUrlSlug'].values):
                    print('Notebook ', name, ' already downloaded')
                    self._nb_identifiers = self._nb_identifiers.loc[~(
                            (self._nb_identifiers['UserName'] == split[0]) &
                            (self._nb_identifiers['CurrentUrlSlug'] == split[1]))]
                else:  # remove the notebook
                    #print('Removing notebook', name, ' not found in db')
                    #path.unlink()
                    pass

            else:  # remove the notebook
                #print('Removing notebook', name, ' not valid')
                #path.unlink()
                pass

    def _http_download(self, skip=0):
        """
        This method implements the HTTP download strategy.
        """
        self._n_successful_downloads = 0
        self._n_failed_downloads = 0
        idx = -1

        for row in tqdm(self._nb_identifiers.itertuples(), total=self._nb_identifiers.shape[0]):
            print("-----")
            print(row[1])
            print(row[2])
            print(row[3])
            filename = re.sub(r'[+\'"?#$%^*@!/\\`~|:;\]\[-]+', '', f'{row[1]}_{row[2]}')
            print(filename)
            print("-----")
            download_path = self._nb_archive_path + f'/{filename}.ipynb'
            if os.path.exists(download_path):
                self._n_successful_downloads += 1
                continue

            # Generate URL
            url = 'https://www.kaggle.com/kernels/scriptcontent/{}/download'.format(row[3])

            # Download notebook content to memory
            # noinspection PyBroadException
            try:
                cookies = {
                    '.ASPXAUTH': 'CD65DB83B31EEED511B518610C82E13397E96A7C2CA4A0175D4F0A0DE73957E1CD713389573DCE8E29DE4B918056C67C79AA325C5251C8258FD72A2AAD177DDD9D6F8121AF335A4F86E2855CB46A52273C858595',
                    'CLIENT_TOKEN': 'eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJ6aGVuZ21pbnlhbmciLCJuYnQiOiIyMDIxLTA3LTA5VDEzOjA3OjIzLjExODY4NjVaIiwiaWF0IjoiMjAyMS0wNy0wOVQxMzowNzoyMy4xMTg2ODY1WiIsImp0aSI6ImNiNmQwNjUyLThlZjktNGRiNy1iYzQ4LTNmYTllMTY1NjE4NyIsImV4cCI6IjIwMjEtMDgtMDlUMTM6MDc6MjMuMTE4Njg2NVoiLCJ1aWQiOjY0OTk2NTQsImZmIjpbIkRvY2tlck1vZGFsU2VsZWN0b3IiLCJBY3RpdmVFdmVudHMiLCJHY2xvdWRLZXJuZWxJbnRlZyIsIktlcm5lbEVkaXRvckNvcmdpTW9kZSIsIkNhaXBFeHBvcnQiLCJDYWlwTnVkZ2UiLCJLZXJuZWxzRmlyZWJhc2VMb25nUG9sbGluZyIsIktlcm5lbHNQcmV2ZW50U3RvcHBlZFRvU3RhcnRpbmdUcmFuc2l0aW9uIiwiS2VybmVsc1BvbGxRdW90YSIsIktlcm5lbHNRdW90YU1vZGFscyIsIkRhdGFzZXRzRGF0YUV4cGxvcmVyVjNUcmVlTGVmdCIsIkF2YXRhclByb2ZpbGVQcmV2aWV3IiwiRGF0YXNldHNEYXRhRXhwbG9yZXJWM0NoZWNrRm9yVXBkYXRlcyIsIkRhdGFzZXRzRGF0YUV4cGxvcmVyVjNDaGVja0ZvclVwZGF0ZXNJbkJhY2tncm91bmQiLCJLZXJuZWxzU3RhY2tPdmVyZmxvd1NlYXJjaCIsIktlcm5lbHNNYXRlcmlhbExpc3RpbmciLCJEYXRhc2V0c01hdGVyaWFsRGV0YWlsIiwiRGF0YXNldHNNYXRlcmlhbExpc3RDb21wb25lbnQiLCJDb21wZXRpdGlvbkRhdGFzZXRzIiwiRGlzY3Vzc2lvbnNVcHZvdGVTcGFtV2FybmluZyIsIlRhZ3NMZWFybkFuZERpc2N1c3Npb25zVUkiLCJOb1JlbG9hZEV4cGVyaW1lbnQiLCJOb3RlYm9va3NMYW5kaW5nUGFnZSIsIkRhdGFzZXRzRnJvbUdjcyIsIlRQVUNvbW1pdFNjaGVkdWxpbmciLCJDb21taXRTY2hlZHVsaW5nIiwiRW1haWxTaWdudXBOdWRnZXMiLCJLZXJuZWxzTGVzc1JhcGlkQXV0b1NhdmUiLCJMZWFybkxhbmRpbmdLTSIsIkJvb2ttYXJrc1VJIiwiQm9va21hcmtzQ29tcHNVSSIsIkNvbXBldGl0aW9uc0ttTGFuZGluZyIsIktlcm5lbFZpZXdlckhpZGVGYWtlRXhpdExvZ1RpbWUiLCJEYXRhc2V0TGFuZGluZ1BhZ2VSb3RhdGluZ1NoZWx2ZXMiLCJMZWFybkxhbmRpbmdEZWZhdWx0TGlzdCIsIkxvd2VyRGF0YXNldEhlYWRlckltYWdlTWluUmVzIiwiTmV3RGlzY3Vzc2lvbnNMYW5kaW5nIiwiTmV3RGlzY3Vzc2lvbnNMYW5kaW5nMkxpbmUiLCJEaXNjdXNzaW9uTGlzdGluZ0ltcHJvdmVtZW50cyIsIktNTGVhcm5MYW5kaW5nVGVzdCIsIktNTGVhcm5MYW5kaW5nVGVzdFZlcnNpb25CIl0sInBpZCI6ImthZ2dsZS0xNjE2MDciLCJzdmMiOiJ3ZWItZmUiLCJzZGFrIjoiQUl6YVN5QTRlTnFVZFJSc2tKc0NaV1Z6LXFMNjU1WGE1SkVNcmVFIiwiYmxkIjoiZjhjNDIzZmMwYzVlOGJlNjU1YWI5MTQ3NGY2ZTY2M2EwZDcwODRiMCJ9.',
                    'CSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL97tIVoIPtTGWtRoHNqCnPabvsqC33TcUIdE7EV2TsfI7RB7GcZqjC8q06L6o_hK5KFqi5pvL4U3Kr4t9-51TV5dLO7kGbcUdcYURbHki53UkxUcMCrED-7ACQOYj_iJo8',
                    'GCLB': 'CMHN55bZ8Y7s5AE',
                    'XSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL9LpNulqfj49z0t-c_lhr-zmvjkWakcjkdRrADuCAOkU5ZbX7a_6G7Ykb1_bMV-Xdaf6nAfBWD_lznX0b2Hd-Y7ZclRCYqr23jlvBrub9i1OMrpC3Govpg4N3h4KwOUvcz1kHoUhuYmoMEP7pYJ1vVD57-sMnSMzoS8p_pGDBQ5hQ',
                    'ka_sessionid': 'd1fc32396914f933968b1a422c2b348d',
                }

                notebook = requests.get(url, allow_redirects=True, timeout=5, cookies=cookies)

            except requests.exceptions.HTTPError:
                logging.exception(f'HTTPError while requesting the notebook at: "{url}"')
                self._n_failed_downloads += 1
                continue

            except Exception:
                logging.exception(f'An error occurred while requesting the notebook at: "{url}"')
                self._n_failed_downloads += 1
                continue

            # Write notebook in folder
            filename = re.sub(r'[+\'"?#$%^*@!/\\`~|:;\]\[-]+', '', f'{row[1]}_{row[2]}')
            download_path = self._nb_archive_path + f'/{filename}.ipynb'
            print(download_path)

            with open(Path(download_path), 'wb') as notebook_file:
                notebook_file.write(notebook.content)

            self._n_successful_downloads += 1
            logging.info(f'Downloaded {row[1]}/{row[2]} (ID: {row[3]})')

            # Wait a bit to avoid a potential IP banning
            time.sleep(10)

    def _api_download(self):
        """
        This method implements the API download strategy.
        """

        # Initialization and authentication
        # It's need kaggle.json token in ~/.kaggle
        api = KaggleApi()
        api.authenticate()

        self._n_successful_downloads = 0
        self._n_failed_downloads = 0

        for row in tqdm(self._nb_identifiers.itertuples(), total=self._nb_identifiers.shape[0]):

            # noinspection PyBroadException
            try:
                api.kernels_pull(f'{row[1]}/{row[2]}', path=Path(self._nb_archive_path))

                # Kaggle API save notebook only with slug name
                # Rename downloaded notebook to username/slug
                nb = Path(self._nb_archive_path + f'/{row[2]}.ipynb')
                nb.rename(self._nb_archive_path + f'/{row[1]}_{row[2]}.ipynb')

            except Exception:
                logging.exception(f'An error occurred while requesting the notebook {row[1]}/{row[2]}')
                self._n_failed_downloads += 1
                continue

            self._n_successful_downloads += 1
            logging.info(f'Downloaded {row[1]}/{row[2]} (ID: {row[3]})')

            # Wait a bit to avoid a potential IP banning
            time.sleep(1)

    def download_notebooks(self, strategy='HTTP', skip=0):
        """
        This method executes the download procedure using the provided strategy after checking the destination folder.

        Args:
            strategy:  The download strategy (``HTTP`` or ``API``). By default it is ``HTTP``.
            skip: the number of notebooks to skip
        """

        self._check_destination_folder()

        # Number of notebooks to download
        total_rows = self._nb_identifiers.shape[0]
        print("Total number of notebooks to download:", total_rows)

        # Wait a bit to ensure the print before tqdm bar
        time.sleep(1)

        # HTTP STRATEGY
        if strategy is 'HTTP':
            self._http_download(skip=skip)

        # API STRATEGY
        elif strategy is 'API':
            self._api_download()

        else:
            raise ValueError("strategy is invalid")

        # Print download session summary
        # Print summary to stdout
        print("Total number of notebooks to download was:", total_rows)
        print("\tNumber of successful downloads:", self._n_successful_downloads)
        print("\tNumber of failed downloads:", self._n_failed_downloads)

        # Print summary to log file
        logging.info('DOWNLOAD COMPLETED.\n'
                     f'Total attempts: {total_rows}:\n'
                     f'\t- {self._n_successful_downloads} successful;\n'
                     f'\t- {self._n_failed_downloads} failed.')


if __name__ == '__main__':
    print(f"## Connecting to {config.db_name} db on port {config.db_port} as user {config.db_username}")
    db_engine = DbCommunicationHandler(config.db_username,
                                       config.db_password,
                                       config.db_host,
                                       config.db_port,
                                       config.db_name)

    print("** QUERING KERNELS TO DOWNLOAD **")
    kernels_ids = db_engine.get_nb_identifiers(config.nb_conf['languages'])

    #downloader = Downloader(kernels_ids.head(num_notebooks), config.nb_archive_path)
    #kernels_ids.head(num_notebooks).to_excel("notebooks_info.xlsx")
    downloader = Downloader(kernels_ids, config.nb_archive_path)
    kernels_ids.to_excel("notebooks_info.xlsx")

    strategies = 'HTTP', 'API'

    print("*******************************")
    print("** NOTEBOOK DOWNLOAD STARTED **")
    print("*******************************")
    print(f'# Selected strategy. {strategies[0]}')
    downloader.download_notebooks(strategy=strategies[0], skip=len(glob.glob1(os.getcwd()+"\\notebooks\\","*.ipynb")))
    print('## Download finished.')
