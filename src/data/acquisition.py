from pathlib import Path

import kagglehub

class KaggleDatasetDownloader:
    """
    Download a file from Kaggle dataset
    
    Parameters:
        dataset_handle:
            kaggle dataset handle in the form of 'owner/dataset-name'
        file_name:
            name of the file to download from the dataset
        destination_dir:
            local destination directory where the downloaded file will be stored
    """
    def __init__(
            self,
            dataset_handle: str,
            file_name: str,
            destination_dir: Path | str
    ) -> None:
        # attributes
        self.dataset_handle = dataset_handle
        self.file_name = file_name
        self.destination_dir = Path(destination_dir)

    ### helper functions
    def _validate_configuration(self) -> None:
        """ Validate downloader configuration """
        # dataset_handle validation
        if not self.dataset_handle.strip():
            raise ValueError('dataset_handle cannot be empty!')

        # file_name validation
        if not self.file_name.strip():
            raise ValueError('file_name cannot be empty!')

    def _resolve_downloaded_file(
            self,
            downloaded_location: Path,
            expected_path: Path
    ) -> Path:
        """ 
        Resolve the downloaded file from KaggleHub's result 
        
        Parameters:
            downloaded_location:
                the path returned by KaggleHub
            expected_path:
                the path where the application expects the file
        """
        # check whether the file exists in the expected_path
        if expected_path.is_file():
            return expected_path

        # check whether KaggleHub returned a file
        if downloaded_location.is_file():
            return downloaded_location

        # check whether KaggleHub returned a directory
        if downloaded_location.is_dir():
            # recursively search for matching files with self.file_name
            matches = list(
                downloaded_location.rglob(self.file_name)
            )

            if len(matches) == 1:
                return matches[0]

            if len(matches) > 1:
                raise RuntimeError(
                    f'Multiple files named {self.file_name!r} were found after downloading!'
                )

        # if no usable file found
        raise FileNotFoundError(
            f'KaggleHub completed without producing the expected file: {expected_path}'
        )

    ### public method
    def download(
            self,
            force: bool = False
    ) -> Path:
        """  
        Download the configured Kaggle dataset file

        If the expected file already exists and force = False, then the existing file is returned        
        """
        # validate the configuration
        self._validate_configuration()

        # create the destination directory
        self.destination_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        # expected file path by the application
        expected_path = self.destination_dir / self.file_name

        # return the file path if it is stored in the expected path
        if expected_path.is_file() and not force:
            return expected_path

        try:
            downloaded_location = Path(
                kagglehub.dataset_download(
                    handle = self.dataset_handle,
                    path = self.file_name,
                    output_dir = str(self.destination_dir),
                    force_download = force
                )
            )
        except Exception as e:
            raise RuntimeError('The Kaggle dataset could not be downloaded!') from e

        # return the resolved downloaded KaggleHub result
        return self._resolve_downloaded_file(
            downloaded_location = downloaded_location,
            expected_path = expected_path
        )