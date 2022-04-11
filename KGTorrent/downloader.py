"""
This module defines the class that handles the actual download of Jupyter notebooks from Kaggle.
"""

import logging
import time
from pathlib import Path
from tqdm import tqdm

import requests
# from kaggle.api.kaggle_api_extended import KaggleApi

# Imports for testing
# import KGTorrent.config as config
# from KGTorrent.db_communication_handler import DbCommunicationHandler
import config as config

from db_communication_handler import DbCommunicationHandler


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
                print("self._nb_identifiers:" + str(self._nb_identifiers.keys())) 
                if (split[0] in self._nb_identifiers['UserName'].values) & \
                        (split[1] in self._nb_identifiers['CurrentUrlSlug'].values):
                    print('Notebook ', name, ' already downloaded')
                    self._nb_identifiers = self._nb_identifiers.loc[~(
                            (self._nb_identifiers['UserName'] == split[0]) &
                            (self._nb_identifiers['CurrentUrlSlug'] == split[1]))]
                else:  # remove the notebook
                    print('Removing notebook', name, ' not found in db')
                    path.unlink()

            else:  # remove the notebook
                print('Removing notebook', name, ' not valid')
                path.unlink()

    def _http_download(self):
        """
        This method implements the HTTP download strategy.
        """
        self._n_successful_downloads = 0
        self._n_failed_downloads = 0

        for row in tqdm(self._nb_identifiers.itertuples(), total=self._nb_identifiers.shape[0]):

            # Generate URL
            url = 'https://www.kaggle.com/kernels/scriptcontent/{}/download'.format(row[3])

            # Download notebook content to memory
            # noinspection PyBroadException
            try:
                # notebook = requests.get(url, allow_redirects=True, timeout=5)

                cookies = {
                    '.ASPXAUTH': 'EBFBA39C84DEC4C1C74F847C55A7477EF4BF3BD3EA959747B274DCCB9992D46CEA57C1477A71DE100F358330F21224BA7895117417551B32EFE07CAA2B43E48C203E0C3E53F63CADC2CD6D1866F7366F624CDC66',
                    'CLIENT_TOKEN': 'eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJ0aW5heWFuZzMwMjQiLCJuYnQiOiIyMDIyLTAzLTEwVDE3OjMyOjUwLjIyOTkyODVaIiwiaWF0IjoiMjAyMi0wMy0xMFQxNzozMjo1MC4yMjk5Mjg1WiIsImp0aSI6IjI5M2VmYzY4LTJhOTQtNGZkNy05ZGViLTk1YTI2M2QyNmNkMyIsImV4cCI6IjIwMjItMDQtMTBUMTc6MzI6NTAuMjI5OTI4NVoiLCJ1aWQiOjk0MjQ3MjcsImRpc3BsYXlOYW1lIjoidGluYXlhbmczMDI0IiwiZW1haWwiOiJ0aW5heWFuZzMwMjRAZ21haWwuY29tIiwidGllciI6Ik5vdmljZSIsInZlcmlmaWVkIjp0cnVlLCJwcm9maWxlVXJsIjoiL3RpbmF5YW5nMzAyNCIsInRodW1ibmFpbFVybCI6Imh0dHBzOi8vc3RvcmFnZS5nb29nbGVhcGlzLmNvbS9rYWdnbGUtYXZhdGFycy90aHVtYm5haWxzL2RlZmF1bHQtdGh1bWIucG5nIiwiZmYiOlsiS2VybmVsc1NhdmVUb0dpdEh1YiIsIkdjbG91ZEtlcm5lbEludGVnIiwiS2VybmVsRWRpdG9yS2l0dHlNb2RlIiwiQ2FpcEV4cG9ydCIsIkNhaXBOdWRnZSIsIktlcm5lbHNGaXJlYmFzZUxvbmdQb2xsaW5nIiwiS2VybmVsc1BvbGxRdW90YSIsIkRhdGFzZXRzRGF0YUV4cGxvcmVyVjNDaGVja0ZvclVwZGF0ZXMiLCJEYXRhc2V0c0RhdGFFeHBsb3JlclYzQ2hlY2tGb3JVcGRhdGVzSW5CYWNrZ3JvdW5kIiwiS2VybmVsc1N0YWNrT3ZlcmZsb3dTZWFyY2giLCJLZXJuZWxzTWF0ZXJpYWxMaXN0aW5nIiwiS2VybmVsc0VtcHR5U3RhdGUiLCJLZXJuZWxWaWV3ZXJSZXF1ZXN0U291cmNlRnJvbUZyb250ZW5kIiwiRGF0YXNldHNNYXRlcmlhbExpc3RDb21wb25lbnQiLCJEYXRhc2V0c1NoYXJlZFdpdGhZb3UiLCJDb21wZXRpdGlvbkRhdGFzZXRzIiwiVFBVQ29tbWl0U2NoZWR1bGluZyIsIktlcm5lbEVkaXRvckZvcmNlVGhlbWVTeW5jIiwiS01MZWFybkRldGFpbCIsIkZyb250ZW5kRXJyb3JSZXBvcnRpbmciLCJGcm9udGVuZENvbnNvbGVFcnJvclJlcG9ydGluZyIsIkxvd2VyRGF0YXNldEhlYWRlckltYWdlTWluUmVzIiwiRGlzY3Vzc2lvbkVtcHR5U3RhdGUiLCJGaWx0ZXJGb3J1bUltYWdlcyIsIlBob25lVmVyaWZ5Rm9yQ29tbWVudHMiLCJQaG9uZVZlcmlmeUZvck5ld1RvcGljIiwiTmV3TmF2QmVoYXZpb3IiLCJOZXdOYXZVc2VyTGlua3MiLCJOZXdLbUxlYWRlcmJvYXJkIiwiSW5DbGFzc1RvQ29tbXVuaXR5UGFnZXMiLCJQaG9uZVZlcmlmaWNhdGlvbk9sZEZsb3dzTmV3VWkiXSwiZmZkIjp7Iktlcm5lbEVkaXRvckF1dG9zYXZlVGhyb3R0bGVNcyI6IjMwMDAwIiwiRnJvbnRlbmRFcnJvclJlcG9ydGluZ1NhbXBsZVJhdGUiOiIwLjAxIiwiRW1lcmdlbmN5QWxlcnRCYW5uZXIiOiJ7XCJiYW5uZXJzXCI6IFt7XCJ1cmlQYXRoUmVnZXhcIjogXCJeKC8uKilcIiwgXCJtZXNzYWdlSHRtbFwiOiBcIkthZ2dsZSByZWNlbnRseSBlbmFjdGVkIGEgbm90ZWJvb2tzIGFidXNlIGNvdW50ZXJtZWFzdXJlIHdoaWNoIHJlc3VsdGVkIGluIGEgbGFyZ2UgbnVtYmVyIG9mIGZhbHNlIGFjY291bnQgYmxvY2tzLiBXZSBoYXZlIGNvcnJlY3RlZCB0aGUgaXNzdWUgYW5kIGFyZSB3b3JraW5nIG9uIHVuYmxvY2tpbmcgdGhlIGFmZmVjdGVkIGFjY291bnRzLiBJdCBpcyBzYWZlIHRvIHJ1biBub3RlYm9va3MgYWdhaW4gYW5kIHdlIGFwb2xvZ2l6ZSBmb3IgdGhlIGluY29udmVuaWVuY2UuIDxhIGhyZWY9XFxcIi9wcm9kdWN0LWZlZWRiYWNrLzI5OTk4OFxcXCI-U2VlIGhlcmUgZm9yIG1vcmUgaW5mbzwvYT5cIiwgIFwiYmFubmVySWRcIjogXCIyMDIyLTAxLTEwLWtlcm5lbC1mYWxzZS1wb3NpdGl2ZS1hYnVzZVwiIH0gXSB9IiwiQ2xpZW50UnBjUmF0ZUxpbWl0IjoiNDAiLCJGZWF0dXJlZENvbW11bml0eUNvbXBldGl0aW9ucyI6IjMzNjExLDMzNjg5LDM0MTg5IiwiQWRkRmVhdHVyZUZsYWdzVG9QYWdlTG9hZFRhZyI6ImRhdGFzZXRzTWF0ZXJpYWxEZXRhaWwifSwicGlkIjoia2FnZ2xlLTE2MTYwNyIsInN2YyI6IndlYi1mZSIsInNkYWsiOiJBSXphU3lBNGVOcVVkUlJza0pzQ1pXVnotcUw2NTVYYTVKRU1yZUUiLCJibGQiOiJhMmNiYWI4MjU0MTk4YzEzNWIyZTkzZTBkOTlmNDA3MWJjNWMwNjZjIn0.',
                    'CSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL8jeVqwKA97sY801PYVdrfeJQRxYSZiVQkDslP_e7v__Y8KQPR7AYxoUsOTreDnZWqcvXNWipjBpyuCm_-sLAc_k3KUdIyPECQe7_jOix7_KDE4rbosL3tIA9YkiAu2sdc',
                    'GCLB': 'CNWm97e3kvSOHA',
                    'XSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL8a33Fc6Q7CRzvKQWBOHF6AWx1McsE7Ro6Nj2bVINV7sxGcmBrWdvwhOF5CM-3AL1cm4AxXLoEG7ZRnKUwQnUywWZGhkkDyC6F1azmAalEeruWp_GPSJde8-819unI8K2iWojwSbfsX_h5D03KnSpc5bl3ZbuXKbX5OKcgZ9BAaig',
                    'ka_sessionid': 'f2f02568fe8268d4531360cb89243c50',
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
            try:
                with open(Path(download_path), 'wb') as notebook_file:
                    notebook_file.write(notebook.content)
            except Exception:
                print("path invalid: " + str(Path(download_path)))
                self._n_failed_downloads += 1
                print("skipped... ")
                continue

            self._n_successful_downloads += 1
            logging.info(f'Downloaded {row[1]}/{row[2]} (ID: {row[3]})')

            # Wait a bit to avoid a potential IP banning
            time.sleep(1)

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

    def download_notebooks(self, strategy='HTTP'):
        """
        This method executes the download procedure using the provided strategy after checking the destination folder.

        Args:
            strategy:  The download strategy (``HTTP`` or ``API``). By default it is ``HTTP``.
        """

        self._check_destination_folder()

        # Number of notebooks to download
        total_rows = self._nb_identifiers.shape[0]
        print("Total number of notebooks to download:", total_rows)

        # Wait a bit to ensure the print before tqdm bar
        time.sleep(1)

        # HTTP STRATEGY
        if strategy == 'HTTP':
            self._http_download()

        # API STRATEGY
        if strategy == 'API':
            # self._api_download()
            print("nope:(")
            exit()

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
    print("** kernels_ids **")

    downloader = Downloader(kernels_ids.head(200), config.nb_archive_path)
    print("** downloader **")
    strategies = 'HTTP', 'API'

    print("*******************************")
    print("** NOTEBOOK DOWNLOAD STARTED **")
    print("*******************************")
    print(f'# Selected strategy. {strategies[0]}')
    downloader.download_notebooks(strategy=strategies[0])
    print('## Download finished.')
