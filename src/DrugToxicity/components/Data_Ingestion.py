import os
import urllib.request as request
import zipfile
from src.DrugToxicity.utils.common import logger, get_size
from src.DrugToxicity.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """Downloads zip from source_URL only if final_data_file does not exist."""
        if self.config.final_data_file.exists():
            logger.info(
                f"Dataset already exists at {self.config.final_data_file} "
                f"({get_size(self.config.final_data_file)}) — skipping download."
            )
            return

        os.makedirs(self.config.root_dir, exist_ok=True)

        if not self.config.local_data_file.exists():
            logger.info(f"Downloading from {self.config.source_URL} ...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"Download complete → {filename}")
            logger.debug(f"Headers: {headers}")
        else:
            logger.info(
                f"Zip already present at {self.config.local_data_file} "
                f"({get_size(self.config.local_data_file)}) — skipping download."
            )

    def extract_zip_file(self):
        """Extracts zip only if final_data_file does not exist."""
        if self.config.final_data_file.exists():
            logger.info("Extracted file already exists — skipping extraction.")
            return

        logger.info(
            f"Extracting {self.config.local_data_file} → {self.config.unzip_dir}"
        )
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info(f"Successfully extracted to {self.config.unzip_dir}")