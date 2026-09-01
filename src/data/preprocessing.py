from pathlib import Path

import polars as pl

from config import (
    SAML_D_RAW_PATH,
    SAML_D_INTERIM_PATH
)

from src.data.loader import SAMLDDataLoader
from src.data.schema import (
    SAML_D_COLUMN_RENAME_MAP,
    SAML_D_INTERIM_COLUMNS,
    SAML_D_TIMESTAMP_FORMAT
)
from src.data.validation import SAMLDDataValidator

class SAMLDPreprocessor:
    """
    Standardize the validated raw SAML-D dataset and write it to interim Parquet storage

    SAMLDPreprocessor is responsible for the following data pipeline steps:
        - validate and lazily scan
        - standardize columns
        - create transaction identifier
        - parse timestamp
        - arrange output schema
        - write temporary Parquet
        - validate temoporary Parquet
        - promote to final interim file

    Parameters:
        raw_data_path:
            Path to the raw SAML-D csv file
        interim_data_path:
            Path to the standardized interim Parquet file    
    """
    def __init__(
            self,
            raw_data_path: Path | str = SAML_D_RAW_PATH,
            interim_data_path: Path | str = SAML_D_INTERIM_PATH
    ) -> None:
        # attributes
        self.raw_data_path = Path(raw_data_path)
        self.interim_data_path = Path(interim_data_path)

    def transform(
            self,
            raw_lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Construct the standardized interim transformation

        This method only creates a lazy query plan. It does not materialize the complete dataset.        
        """
        # validate the input schema from raw dataset
        SAMLDDataValidator.validate_schema(
            lazy_frame = raw_lazy_frame
        )

        return (
            raw_lazy_frame
            .with_row_index(
                name = 'transaction_id',
                offset = 0
            )
            .rename(
                SAML_D_COLUMN_RENAME_MAP
            )
            .with_columns(
                [
                    pl.concat_str(
                        [
                            pl.col('date'),
                            pl.lit(' '),
                            pl.col('time')
                        ]
                    )
                    .str.strptime(
                        dtype = pl.Datetime,
                        format = SAML_D_TIMESTAMP_FORMAT,
                        strict = True,
                        exact = True
                    )
                    .alias('timestamp'),
                    pl.col('transaction_id')
                    .cast(
                        pl.UInt64
                    )
                ]
            )
            .drop(
                [
                    'date',
                    'time'
                ]
            )
            .select(
                list(SAML_D_INTERIM_COLUMNS)
            )
        )

    def convert_to_parquet(
            self,
            overwrite: bool = False
    ) -> Path:
        """
        Convert the raw SAML-D csv into standardized Parquet

        An existing valid output is returned unless overwrite=True. The transformation is first written to a temporary file so that 
        a failed write does not damage an existing dataset.        
        """
        # create the interim directory
        self.interim_data_path.parent.mkdir(
            parents = True,
            exist_ok = True
        )

        # handles if the interim path exists
        if self.interim_data_path.exists():
            # check whether interim_data_path is a file
            if not self.interim_data_path.is_file():
                raise ValueError(
                    f'SAML-D interim path exists but is not a file: {self.interim_data_path}'
                )

            # reuse existing output, otherwise lazily scan and validate file and schema
            if not overwrite:
                # validate interim file
                SAMLDDataValidator.validate_interim_file(
                    file_path = self.interim_data_path
                )

                # lazily scan iterim file
                existing_lazy_frame = pl.scan_parquet(
                    source = self.interim_data_path
                )

                # validate interim file schema
                SAMLDDataValidator.validate_interim_schema(
                    lazy_frame = existing_lazy_frame
                )

                return self.interim_data_path 

        # construct the temporary path
        temporary_path = (
            self.interim_data_path
            .with_name(
                f'{self.interim_data_path.stem}'
                '.tmp'
                f'{self.interim_data_path.suffix}'
            )
        )

        # check if temporary_path exists
        if temporary_path.exists():
            # check if temporary_path is a file
            if not temporary_path.is_file():
                raise ValueError(
                    f'Temporary Parquet path exists but is not a file: {temporary_path}'
                )

            # delete the stale temporary file
            temporary_path.unlink()

        # create raw data loader instance
        raw_loader = SAMLDDataLoader(
            raw_data_path = self.raw_data_path
        )

        # lazily scan the raw dataset
        raw_lazy_frame = raw_loader.scan_raw()

        # construct the interim transformation
        interim_lazy_frame = self.transform(
            raw_lazy_frame = raw_lazy_frame
        )

        try:
            # execute the lazy query plan and write to temporary parquet path
            interim_lazy_frame.sink_parquet(
                path = temporary_path,
                compression = 'zstd',
                statistics = True,
                maintain_order = True,
                engine = 'streaming'
            )

            # validate temporary file
            SAMLDDataValidator.validate_interim_file(
                file_path = temporary_path
            )

            # lazily scan the temporary dataset
            temporary_lazy_frame = pl.scan_parquet(
                source = temporary_path
            )

            # validate interim schema for the temporary file
            SAMLDDataValidator.validate_interim_schema(
                lazy_frame = temporary_lazy_frame
            )

            # move the temporary file to the final interim data path
            temporary_path.replace(self.interim_data_path)

        except Exception:
            if temporary_path.is_file():
                # delete any partial or invalid temporary Parquet file
                temporary_path.unlink()

            raise

        return self.interim_data_path