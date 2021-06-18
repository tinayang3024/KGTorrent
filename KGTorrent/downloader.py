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
            idx += 1
            if idx < skip:
                self._n_successful_downloads += 1
                continue

            # Generate URL
            url = 'https://www.kaggle.com/kernels/scriptcontent/{}/download'.format(row[3])

            # Download notebook content to memory
            # noinspection PyBroadException
            try:

                # cookies = {
                #     '.ASPXAUTH': '7516769A1C8A8FE81812467ABC3B801EA35B55F32761F27BA27A2F2AC992F856C07757302E3165502C19EE3E4E232023C9AC85B7E9225F6370D8A971A4E85AF48A142A4B2554053190BEB19B485CD0403DD4F4F8',
                #     'ka_sessionid': 'e179d3cb492015052b50858ff1c529e0aab117c9',
                #     '_ga': 'GA1.2.908532115.1613499167',
                # }

                cookies = {
                    '.ASPXAUTH': '2656A0B5422BFF8AF3E2FE87BBB9D34666563EEA51C5AEBDC483A737B52B1A26A10596ABAB84E815A566896C040999059571590A2391B86FB544FDA27696FC98F9176387B91FC644E4316BC0D1348444DD1C11EF',
                    'CLIENT_TOKEN': 'eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJ6aGVuZ21pbnlhbmciLCJuYnQiOiIyMDIxLTA2LTE3VDE4OjU2OjUwLjY0MjEwMjRaIiwiaWF0IjoiMjAyMS0wNi0xN1QxODo1Njo1MC42NDIxMDI0WiIsImp0aSI6IjU0YWNiYjU1LTUxMjAtNDljMi04ZGMwLWYzMDgwZGJiZTBiNyIsImV4cCI6IjIwMjEtMDctMTdUMTg6NTY6NTAuNjQyMTAyNFoiLCJ1aWQiOjY0OTk2NTQsImZmIjpbIkRvY2tlck1vZGFsU2VsZWN0b3IiLCJBY3RpdmVFdmVudHMiLCJHY2xvdWRLZXJuZWxJbnRlZyIsIktlcm5lbEVkaXRvckNvcmdpTW9kZSIsIkNhaXBFeHBvcnQiLCJDYWlwTnVkZ2UiLCJLZXJuZWxzRmlyZWJhc2VMb25nUG9sbGluZyIsIktlcm5lbHNQcmV2ZW50U3RvcHBlZFRvU3RhcnRpbmdUcmFuc2l0aW9uIiwiS2VybmVsc1BvbGxRdW90YSIsIktlcm5lbHNRdW90YU1vZGFscyIsIkRhdGFzZXRzRGF0YUV4cGxvcmVyVjNUcmVlTGVmdCIsIkF2YXRhclByb2ZpbGVQcmV2aWV3IiwiRGF0YXNldHNEYXRhRXhwbG9yZXJWM0NoZWNrRm9yVXBkYXRlcyIsIkRhdGFzZXRzRGF0YUV4cGxvcmVyVjNDaGVja0ZvclVwZGF0ZXNJbkJhY2tncm91bmQiLCJLZXJuZWxzU3RhY2tPdmVyZmxvd1NlYXJjaCIsIktlcm5lbHNNYXRlcmlhbExpc3RpbmciLCJEYXRhc2V0c01hdGVyaWFsRGV0YWlsIiwiRGF0YXNldHNNYXRlcmlhbExpc3RDb21wb25lbnQiLCJDb21wZXRpdGlvbkRhdGFzZXRzIiwiRGlzY3Vzc2lvbnNVcHZvdGVTcGFtV2FybmluZyIsIlRhZ3NFeHBlcmltZW50VUkiLCJUYWdzTGVhcm5BbmREaXNjdXNzaW9uc1VJIiwiTm9SZWxvYWRFeHBlcmltZW50IiwiTm90ZWJvb2tzTGFuZGluZ1BhZ2UiLCJEYXRhc2V0c0Zyb21HY3MiLCJLZXJuZWxzTGVzc1JhcGlkQXV0b1NhdmUiLCJMZWFybkxhbmRpbmdLTSIsIkJvb2ttYXJrc1VJIiwiQm9va21hcmtzQ29tcHNVSSIsIkNvbXBldGl0aW9uc0ttTGFuZGluZyIsIktlcm5lbFZpZXdlckhpZGVGYWtlRXhpdExvZ1RpbWUiLCJEYXRhc2V0TGFuZGluZ1BhZ2VSb3RhdGluZ1NoZWx2ZXMiLCJMZWFybkxhbmRpbmdEZWZhdWx0TGlzdCIsIkxvd2VyRGF0YXNldEhlYWRlckltYWdlTWluUmVzIiwiS01MZWFybkxhbmRpbmdUZXN0IiwiS01MZWFybkxhbmRpbmdUZXN0VmVyc2lvbkIiXSwicGlkIjoia2FnZ2xlLTE2MTYwNyIsInN2YyI6IndlYi1mZSIsInNkYWsiOiJBSXphU3lBNGVOcVVkUlJza0pzQ1pXVnotcUw2NTVYYTVKRU1yZUUiLCJibGQiOiJkYmRlYTQyODYwMjM4ZTZiY2EwM2QyMTUzZmYzZTdmMTc3OWNmNDYxIn0.',
                    'CSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL-GSFUhhm55gMh4qVwgoN6VcDzKRJu3BLX1PwHQj_jDW3PMQCeYY5W5CPG95xBYbpByaLDMttyWHqKMy2HSbPXCKlRnRyFir3hjJnDwK3uBjCIGLWb05E6MmbE84eYkQC4',
                    'GCLB': 'CJ60sbjq3tGpUA',
                    'XSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL8vmotCo64cSevITk9iGUEAclAfItyAq4mjPpRL4if23Ly-EPLqU3XnnSwB3_dOwR1nqM77PGzNnD5cqKPoiHIAM1Jl8LDu0UJulcu7XjsRJEwnHI-DD4JAzoPJJ3RIQeemnCqryHRYpJLSSSb_eQcqvDXaMbGSPw0yS7nHkHqYbA',
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
            download_path = self._nb_archive_path + f'/{row[1]}_{row[2]}.ipynb'
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
    num_repos = 10
    print(f"## Connecting to {config.db_name} db on port {config.db_port} as user {config.db_username}")
    db_engine = DbCommunicationHandler(config.db_username,
                                       config.db_password,
                                       config.db_host,
                                       config.db_port,
                                       config.db_name)

    print("** QUERING KERNELS TO DOWNLOAD **")
    kernels_ids = db_engine.get_nb_identifiers(config.nb_conf['languages'], num_notebooks=num_repos)

    #downloader = Downloader(kernels_ids.head(num_notebooks), config.nb_archive_path)
    #kernels_ids.head(num_notebooks).to_excel("notebooks_info.xlsx")
    downloader = Downloader(kernels_ids, config.nb_archive_path)
    kernels_ids.to_excel("notebooks_info.xlsx")

    strategies = 'HTTP', 'API'
    #strategies = 'API', 'HTTP'

    print("*******************************")
    print("** NOTEBOOK DOWNLOAD STARTED **")
    print("*******************************")
    print(f'# Selected strategy. {strategies[0]}')
    downloader.download_notebooks(strategy=strategies[0], skip=len(glob.glob1(os.getcwd(),"*.ipynb")))
    print('## Download finished.')
