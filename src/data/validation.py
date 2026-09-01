import polars as pl
from pathlib import Path

from src.data.schema import (
    SAML_D_REQUIRED_COLUMNS,
    SAML_D_SCHEMA_OVERRIDES,
    SAML_D_TIMESTAMP_FORMAT,
    SAML_D_INTERIM_COLUMNS,
    SAML_D_INTERIM_SCHEMA
)

class SAMLDDataValidator:
    """ A utility class for validating the raw SAML-D file, schema and content """
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
        """ Validate that all required SAML-D columns and data types are present """
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

        # validate data types
        invalid_dtypes: list[str] = []

        # actual schema
        actual_schema = lazy_frame.collect_schema()

        for column_name, expected_dtype in SAML_D_SCHEMA_OVERRIDES.items():
            # actual dtype of a column in the schema
            actual_dtype = actual_schema[column_name]

            # check whether actual vs. expected dtypes are matching
            if actual_dtype != expected_dtype:
                invalid_dtypes.append(
                    f'{column_name}: expected {expected_dtype}, '
                    f'received {actual_dtype}'
                )

        # sorted invalid dtypes if present
        if invalid_dtypes:
            sorted_invalid_dtypes = '; '.join(
                sorted(invalid_dtypes)
            )

            raise ValueError(
                'SAML-D data type validation failed. '
                f'Invalid dtypes: {sorted_invalid_dtypes}'
            )

    @staticmethod
    def _timestamp_expression() -> pl.Expr:
        """
        Private helper to construct a parsed timestamp expression from the raw Date and Time columns
        
        strict = False converts invalid values to null so that all parsing failures can be counted during validation
        """
        return (
            pl.concat_str(
                [
                    pl.col('Date'),
                    pl.lit(' '),
                    pl.col('Time')
                ]
            )
            .str.strptime(
                dtype = pl.Datetime,
                format = SAML_D_TIMESTAMP_FORMAT,
                strict = False,
                exact = True
            )
        )

    @staticmethod
    def get_content_summary(lazy_frame: pl.LazyFrame) -> dict[str, int]:
        """
        Scan the complete raw dataset and return aggregated validation metrics

        The result contains only aggregated values, so the complete dataset is not materialized in memory
        """
        # validate the SAML-D schema for expected columns and dtypes
        SAMLDDataValidator.validate_schema(
            lazy_frame = lazy_frame
        )

        # dynamically calculate null counts per expected column in the schema
        null_count_expressions = [
            pl.col(column_name)
            .null_count()
            .alias(
                f'{column_name.lower()}_null_count'
            )
            for column_name in SAML_D_REQUIRED_COLUMNS
        ]

        # summary aggregated validation metrics
        summary_frame = (
            lazy_frame
            .select(
                [
                    # total row counts
                    pl.len().alias(
                        'row_count'
                    ),
                    # unpacking null counts per column
                    *null_count_expressions,
                    # count empty sender accounts
                    (
                        (
                            pl.col('Sender_account')
                            .str.strip_chars()
                            == ''
                        )
                        .fill_null(False)
                        .sum()
                        .alias(
                            'empty_sender_account_count'
                        )
                    ),
                    # count empty receiver accounts
                    (
                        (
                            pl.col('Receiver_account')
                            .str.strip_chars()
                            == ''
                        )
                        .fill_null(False)
                        .sum()
                        .alias(
                            'empty_receiver_account_count'
                        )
                    ),
                    # count NaN amounts
                    (
                        pl.col('Amount')
                        .is_nan()
                        .fill_null(False)
                        .sum()
                        .alias(
                            'nan_amount_count'
                        )
                    ),
                    # count infinite amounts
                    (
                        pl.col('Amount')
                        .is_infinite()
                        .fill_null(False)
                        .sum()
                        .alias(
                            'infinite_amount_count'
                        )
                    ),
                    # count non-positive amounts
                    (
                        (
                            pl.col('Amount') <= 0
                        )
                        .fill_null(False)
                        .sum()
                        .alias(
                            'non_positive_amount_count'
                        )
                    ),
                    # count invalid laundering types
                    (
                        (
                            ~(
                                pl.col('Is_laundering')
                                .is_in([0, 1])
                                .fill_null(False)
                            )
                        )
                        .sum()
                        .alias(
                            'invalid_label_count'
                        )
                    ),
                    # count legitimate transactions
                    (
                        (
                            pl.col('Is_laundering') == 0
                        )
                        .fill_null(False)
                        .sum()
                        .alias(
                            'normal_transaction_count'
                        )
                    ),
                    # count non-legitimate transactions
                    (
                        (
                            pl.col('Is_laundering') == 1
                        )
                        .fill_null(False)
                        .sum()
                        .alias(
                            'laundering_transaction_count'
                        )
                    ),
                    # count unparseable timestamps
                    (
                        SAMLDDataValidator
                        ._timestamp_expression()
                        .is_null()
                        .sum()
                        .alias(
                            'unparseable_timestamp_count'
                        )
                    ),
                    # count self-transfers
                    (
                        (
                            pl.col('Sender_account')
                            == pl.col('Receiver_account')
                        )
                        .fill_null(False)
                        .sum()
                        .alias(
                            'self_transfer_count'
                        )
                    )
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

        # convert one-row DataFrame into dictionary 
        summary_row = summary_frame.row(
            index = 0,
            named = True
        )

        return {
            metric_name: int(metric_value)
            for metric_name, metric_value
            in summary_row.items()
        }

    @staticmethod
    def validate_content(lazy_frame: pl.LazyFrame) -> dict[str, int]:
        """
        Validate critical content rules across the complete dataset

        A content summary is returned when all critical validation rules pass        
        """
        # get aggregated validation metrics
        summary = (
            SAMLDDataValidator
            .get_content_summary(
                lazy_frame = lazy_frame
            )
        )

        # check whether the dataset has no records
        if summary['row_count'] == 0:
            raise ValueError(
                'SAML-D content validation failed: the dataset containes no records!'
            )

        # collect validation failures
        validation_errors: list[str] = []

        # defining columns that must include no null values
        critical_null_columns = [
            'Time',
            'Date',
            'Sender_account',
            'Receiver_account',
            'Amount'
        ]

        for col_name in critical_null_columns:
            # append non-negative null count to validation_errors
            metric_name = f'{col_name.lower()}_null_count'

            null_count = summary[metric_name]

            if null_count > 0:
                validation_errors.append(
                    f'{col_name} contains '
                    f'{null_count:,} null values'
                )

        # check empty sender accounts
        if summary['empty_sender_account_count'] > 0:
            validation_errors.append(
                'Sender_account contains '
                f'{summary['empty_sender_account_count']:,} empty values'
            )

        # check empty receiver accounts
        if summary['empty_receiver_account_count'] > 0:
            validation_errors.append(
                'Receiver_account contains '
                f'{summary['empty_receiver_account_count']:,} empty values'
            )

        # check NaN amounts
        if summary['nan_amount_count'] > 0:
            validation_errors.append(
                'Amount contains '
                f'{summary['nan_amount_count']:,} NaN values'
            )

        # check infinite amounts
        if summary['infinite_amount_count'] > 0:
            validation_errors.append(
                'Amount contains '
                f'{summary['infinite_amount_count']:,} infinite values'
            )

        # check invalid labels
        if summary['invalid_label_count'] > 0:
            validation_errors.append(
                'Is_laundering contains '
                f'{summary['invalid_label_count']:,} invalid values'
            )

        # check legitimate transactions
        if summary['normal_transaction_count'] == 0:
            validation_errors.append(
                'the dataset contains no normal transactions'
            )

        # check non-legitimate transactions
        if summary['laundering_transaction_count'] == 0:
            validation_errors.append(
                'the dataset contains no laundering transactions'
            )

        # check unparseable timestamps
        if summary['unparseable_timestamp_count'] > 0:
            validation_errors.append(
                'Date and Time contain '
                f'{summary['unparseable_timestamp_count']:,} unparseable combinations'
            )

        # sorted validations errors if present
        if validation_errors:
            sorted_validation_errors = '; '.join(
                sorted(validation_errors)
            )

            raise ValueError(
                'SAML-D content validation failed. '
                f'{sorted_validation_errors}'
            )

        return summary
    
    @staticmethod
    def validate_interim_file(file_path: Path) -> None:
        """ Validate that the interim Parquet dataset is usable """
        # check whether file_path exists
        if not file_path.exists():
            raise FileNotFoundError(
                f'SAML-D interim file does not exist: {file_path}'
            )

        # check whether file_path is a file
        if not file_path.is_file():
            raise ValueError(
                f'SAML-D interim path is not a file: {file_path}'
            )

        # check whether suffix is valid
        if file_path.suffix.lower() != '.parquet':
            raise ValueError(
                'SAML-D interim file must be Parquet, '
                f'received: {file_path.suffix}'
            )

        # check whether file_path is empty
        if file_path.stat().st_size == 0:
            raise ValueError(
                f'SAML-D interim file is empty: {file_path}'
            )

    @staticmethod
    def validate_interim_schema(lazy_frame: pl.LazyFrame) -> None:
        """ Validate the standardized SML-D Parquet schema, including column order and data types """
        # get the actual schema from interim file
        actual_schema = lazy_frame.collect_schema()

        # actual columns in the interim file
        actual_columns = actual_schema.names()

        # expected columns in the interim dataset
        expected_columns = list(
            SAML_D_INTERIM_COLUMNS
        )

        # check the mismatch between actual and expected columns
        if actual_columns != expected_columns:
            raise ValueError(
                'SAML-D interim colun validation failed. '
                f'Expected: {expected_columns}. '
                f'Received: {actual_columns}.'
            )

        # validate data types
        invalid_dtypes: list[str] = []

        for column_name, expected_dtype in SAML_D_INTERIM_SCHEMA.items():
            # actual dtype of a column in the interim schema
            actual_dtype = actual_schema[column_name]

            # check whether actual vs. expected dtypes are matching
            if actual_dtype != expected_dtype:
                invalid_dtypes.append(
                    f'{column_name}: '
                    f'expected {expected_dtype}, '
                    f'received {actual_dtype}'
                )

        # sorted invalid dtypes if present
        if invalid_dtypes:
            sorted_invalid_dtypes = '; '.join(
                sorted(invalid_dtypes)
            )

            raise ValueError(
                'SAML-D interim data type validation failed. '
                f'{sorted_invalid_dtypes}'
            )