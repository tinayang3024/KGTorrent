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

                # cookies = {
                #     '.ASPXAUTH': '7516769A1C8A8FE81812467ABC3B801EA35B55F32761F27BA27A2F2AC992F856C07757302E3165502C19EE3E4E232023C9AC85B7E9225F6370D8A971A4E85AF48A142A4B2554053190BEB19B485CD0403DD4F4F8',
                #     'ka_sessionid': 'e179d3cb492015052b50858ff1c529e0aab117c9',
                #     '_ga': 'GA1.2.908532115.1613499167',
                # }

                cookies = {
                    '.ASPXAUTH': 'B5F3B128DB6EA47602D87C55FCCEEC1174D936241E39FAD21C339029A8D32C674469A32412B5C2531277BD0CA6D89F93CE1456E4AC0ECFF2DE5DB3C3AF40C3912E002B3B81A2A43ECCD3312125CA1B067BEE3F82',
                    'CLIENT_TOKEN': 'eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJ5YW5nemhtOSIsIm5idCI6IjIwMjEtMDYtMTZUMTM6MDQ6MTkuMzA0NzY1NVoiLCJpYXQiOiIyMDIxLTA2LTE2VDEzOjA0OjE5LjMwNDc2NTVaIiwianRpIjoiYWU2M2UyZWEtZTdjNi00ODU2LTg4MzYtMTBkMDE5NzhlMjFmIiwiZXhwIjoiMjAyMS0wNy0xNlQxMzowNDoxOS4zMDQ3NjU1WiIsInVpZCI6NzcwMTY0NCwiZmYiOlsiRG9ja2VyTW9kYWxTZWxlY3RvciIsIkFjdGl2ZUV2ZW50cyIsIkdjbG91ZEtlcm5lbEludGVnIiwiS2VybmVsRWRpdG9yQ29yZ2lNb2RlIiwiQ2FpcEV4cG9ydCIsIkNhaXBOdWRnZSIsIktlcm5lbHNGaXJlYmFzZUxvbmdQb2xsaW5nIiwiS2VybmVsc1ByZXZlbnRTdG9wcGVkVG9TdGFydGluZ1RyYW5zaXRpb24iLCJLZXJuZWxzUG9sbFF1b3RhIiwiS2VybmVsc1F1b3RhTW9kYWxzIiwiRGF0YXNldHNEYXRhRXhwbG9yZXJWM1RyZWVMZWZ0IiwiQXZhdGFyUHJvZmlsZVByZXZpZXciLCJEYXRhc2V0c0RhdGFFeHBsb3JlclYzQ2hlY2tGb3JVcGRhdGVzIiwiRGF0YXNldHNEYXRhRXhwbG9yZXJWM0NoZWNrRm9yVXBkYXRlc0luQmFja2dyb3VuZCIsIktlcm5lbHNTdGFja092ZXJmbG93U2VhcmNoIiwiS2VybmVsc01hdGVyaWFsTGlzdGluZyIsIkRhdGFzZXRzTWF0ZXJpYWxEZXRhaWwiLCJEYXRhc2V0c01hdGVyaWFsTGlzdENvbXBvbmVudCIsIkNvbXBldGl0aW9uRGF0YXNldHMiLCJEaXNjdXNzaW9uc1Vwdm90ZVNwYW1XYXJuaW5nIiwiVGFnc0V4cGVyaW1lbnRVSSIsIlRhZ3NMZWFybkFuZERpc2N1c3Npb25zVUkiLCJOb1JlbG9hZEV4cGVyaW1lbnQiLCJOb3RlYm9va3NMYW5kaW5nUGFnZSIsIkRhdGFzZXRzRnJvbUdjcyIsIktlcm5lbHNMZXNzUmFwaWRBdXRvU2F2ZSIsIkJvb2ttYXJrc1VJIiwiQ29tcGV0aXRpb25zS21MYW5kaW5nIiwiS2VybmVsVmlld2VySGlkZUZha2VFeGl0TG9nVGltZSIsIkRhdGFzZXRMYW5kaW5nUGFnZVJvdGF0aW5nU2hlbHZlcyIsIkxlYXJuTGFuZGluZ0RlZmF1bHRMaXN0IiwiTG93ZXJEYXRhc2V0SGVhZGVySW1hZ2VNaW5SZXMiLCJLTUxlYXJuTGFuZGluZ1Rlc3QiLCJLTUxlYXJuTGFuZGluZ1Rlc3RWZXJzaW9uQiJdLCJwaWQiOiJrYWdnbGUtMTYxNjA3Iiwic3ZjIjoid2ViLWZlIiwic2RhayI6IkFJemFTeUE0ZU5xVWRSUnNrSnNDWldWei1xTDY1NVhhNUpFTXJlRSIsImJsZCI6IjFjZWJjYWI3MzE3YTliNmE3ZDBlYzFhODcyZjliMjU1YWJiNDk0ZWUifQ.',
                    'CSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL-rc_MPdQcdtoBW_kqIfNB4nb3crEEsu-zP4FDu7eUG0NVjxpJTjKKf26Z4jn4dJPhK3zuzDMBSXBikURsxihBt-BqemwilMYb9OzcYFS_NxMiuxzw8GxMI55o8j5043fM',
                    'GCLB': 'CMD9leLQ89DSWQ',
                    'XSRF-TOKEN': 'CfDJ8LdUzqlsSWBPr4Ce3rb9VL9EXj1aRzsJYexhTpDahviZq6sLZpNuKKdmpYClv_Iy6DRK28QRmLHbYYiw2WnGhX7IrALBkJb0xyeC63nn1E0p3wIkpeHF2O2wWkTtD79THSHoiNKKC7iJDNTACWWwzY2o7ZXHPopMfigvM8RW6rpAfxbs7vu3v_YFMqGQA9ZmeA',
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
        if strategy is 'HTTP':
            self._http_download()

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
    num_notebooks = 10
    print(f"## Connecting to {config.db_name} db on port {config.db_port} as user {config.db_username}")
    db_engine = DbCommunicationHandler(config.db_username,
                                       config.db_password,
                                       config.db_host,
                                       config.db_port,
                                       config.db_name)

    print("** QUERING KERNELS TO DOWNLOAD **")
    kernels_ids = db_engine.get_nb_identifiers(config.nb_conf['languages'], num_notebooks=num_notebooks)

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
    downloader.download_notebooks(strategy=strategies[0])
    print('## Download finished.')
