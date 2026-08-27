import polars as pl
from pathlib import Path

from src.data.schema import SAML_D_REQUIRED_COLUMNS

class SAMLDDataValidator:
    """ Validate the raw SAML-D file and its schema """
    @staticmethod
    def validate_file(file_path: Path) -> None:
        """ Validate that the raw dataset exists and is usable """
        # check whether file_path exists
        if not file_path.exists():
            raise FileNotFoundError(
                f'SAML-D file does not exists: {file_path}'
            )

        # check whether file_path is a file
        if not file_path.is_file():
            raise ValueError(
                f'SAML-D path is not a file: {file_path}'
            )

        # check whether suffix is valid
        if file_path.suffix.lower() != '.csv':
            raise ValueError(
                f'SAML-D raw file must be CSV, received: {file_path.suffix}'
            )

        # check whether file_path is empty
        if file_path.stat().st_size == 0:
            raise ValueError(
                f'SAML-D file is empty: {file_path}'
            )

    @staticmethod
    def validate_schema(lazy_frame: pl.LazyFrame) -> None:
        """ Validate that all required SAML-D columns are present """
        # actual columns in the raw dataset
        actual_cols = set(
            lazy_frame
            .collect_schema()
            .names()
        )

        # required columns in the raw dataset
        required_cols = set(
            SAML_D_REQUIRED_COLUMNS
        )

        # missing columns that are not present in the actual raw dataset
        missing_cols = required_cols - actual_cols

        # sorted missing columns if present
        if missing_cols:
            sorted_missing_cols = ', '.join(
                sorted(missing_cols)
            )

            raise ValueError(
                'SAML-D schema validation failed. '
                f'Missing columns: {sorted_missing_cols}'
            )