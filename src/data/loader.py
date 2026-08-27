import polars as pl
from pathlib import Path

from config import SAML_D_RAW_PATH

from src.data.schema import SAML_D_SCHEMA_OVERRIDES
from src.data.validation import SAMLDDataValidator

class SAMLDDataLoader:
    """ 
    Load and lazily scan the local SAML-D dataset 
    
    Parameters:
        raw_data_path:
            Path to the local raw SAML-D csv file
    """
    def __init__(
            self,
            raw_data_path: Path | str = SAML_D_RAW_PATH
    ) -> None:
        # attributes
        self.raw_data_path = Path(raw_data_path)

    def scan_raw(self) -> pl.LazyFrame:
        """
        Lazily scan the raw SAML-D csv file

        The complete dataset is not loaded into memory
        """
        # validate the raw dataset exists and is usable
        SAMLDDataValidator.validate_file(
            file_path = self.raw_data_path
        )

        # lazy csv scan
        lazy_frame = pl.scan_csv(
            source = self.raw_data_path,
            has_header = True,
            schema_overrides = SAML_D_SCHEMA_OVERRIDES,
            try_parse_dates = False,
            low_memory = True,
            rechunk = False
        )

        # validate all required SAML-D columns are present
        SAMLDDataValidator.validate_schema(
            lazy_frame = lazy_frame
        )

        # return the query plan without materializing the complete dataset
        return lazy_frame

    def load_raw_sample(
            self,
            n_rows: int = 10_000
    ) -> pl.DataFrame:
        """  
        Load the first n_rows of the raw dataset

        load_raw_sample method is intended for inspection and development
        """
        # check the n_rows that should be strictly positive
        if n_rows <= 0:
            raise ValueError(
                'n_rows must be greater than 0'
            )

        # return the materialized complete dataset w.r.t. the query plan (pl.LazyFrame -> pl.DataFrame)
        return (
            self.scan_raw()
            .head(n_rows)
            .collect(engine = 'streaming')
        )