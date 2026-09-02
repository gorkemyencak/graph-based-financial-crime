from collections.abc import Sequence

import polars as pl

from src.data.validation import SAMLDDataValidator

class SAMLDTransactionEDA:
    """ 
    Perform scalable transaction-level and temporal EDA on the standardized SAML-D Parquet dataset 
    
    SAMLDTransactionEDA class keeps the transaction data lazy and materializs only aggregated summaries:
        - schema validation
        - transaction-level summaries
        - temporal summaries
        - categorical summaries
        - account-level diagnostics
        - graph edge diagnostics
        - duplicate analysis
    """

    CATEGORICAL_COLUMNS: tuple[str, ...] = (
        'payment_currency',
        'received_currency',
        'sender_bank_location',
        'receiver_bank_location',
        'payment_type',
        'laundering_type'
    )

    AMOUNT_GROUP_COLUMNS: tuple[str, ...] = (
        'payment_currency',
        'received_currency',
        'payment_type',
        'is_laundering',
        'laundering_type'
    )

    DUPLICATE_KEY_COLUMNS: tuple[str, ...] = (
        'timestamp',
        'sender_account',
        'receiver_account',
        'amount',
        'payment_currency',
        'received_currency',
        'sender_bank_location',
        'receiver_bank_location',
        'payment_type',
        'is_laundering',
        'laundering_type'
    )

    def __init__(
            self,
            transactions: pl.LazyFrame
    ) -> None:
        # validate interim schema
        SAMLDDataValidator.validate_interim_schema(
            lazy_frame = transactions
        )

        # attributes
        self.transactions = transactions

    # helper private methods
    @staticmethod
    def _collect(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
        """ Collect an aggregated LazyFrame using the streaming engine """
        return lazy_frame.collect(
            engine = 'streaming'
        )

    def _build_account_events(self) -> pl.LazyFrame:
        """ Represent sender and receiver appearances uniformly """
        # define sender events
        sender_events = (
            self.transactions
            .select(
                [
                    # account
                    pl.col('sender_account')
                    .alias(
                        'account'
                    ),
                    # bank location
                    pl.col('sender_bank_location')
                    .alias(
                        'bank_location'
                    ),
                    # laundering label
                    pl.col('is_laundering'),
                    # outgoing event
                    pl.lit(
                        1,
                        dtype = pl.UInt8
                    )
                    .alias(
                        'outgoing_event'
                    ),
                    # incoming event
                    pl.lit(
                        0,
                        dtype = pl.UInt8
                    )
                    .alias(
                        'incoming_event'
                    )
                ]
            )
        )

        # define receiver events
        receiver_events = (
            self.transactions
            .select(
                [
                    # account
                    pl.col('receiver_account')
                    .alias(
                        'account'
                    ),
                    # bank location
                    pl.col('receiver_bank_location')
                    .alias(
                        'bank_location'
                    ),
                    # laundering label
                    pl.col('is_laundering'),
                    # outgoing event
                    pl.lit(
                        0,
                        dtype = pl.UInt8
                    )
                    .alias(
                        'outgoing_event'
                    ),
                    # incoming event
                    pl.lit(
                        1,
                        dtype = pl.UInt8
                    )
                    .alias(
                        'incoming_event'
                    )
                ]
            )
        )

        return pl.concat(
            [
                sender_events,
                receiver_events
            ],
            how = 'vertical'
        )

    # transaction EDA public methods
    def get_transaction_overview(self) -> pl.DataFrame:
        """ Return a single-row high-level transaction and date-range statistics """
        # define high-level overview query on the transaction dataset
        overview_query = (
            self.transactions
            .select(
                [   
                    # total transaction count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'transaction_count'
                    ),
                    # unique transaction count
                    pl.col('transaction_id')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'unique_transaction_id_count'
                    ),
                    # unique sender count
                    pl.col('sender_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'unique_sender_account'
                    ),
                    # unique receiver account
                    pl.col('receiver_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'unique_receiver_account'
                    ),
                    # legitimate transaction count
                    (
                        pl.col('is_laundering') == 0
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'normal_transaction_count'
                    ),
                    # non-legitimate transaction count
                    (
                        pl.col('is_laundering') == 1
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'laundering_transaction_count'
                    ),
                    # minimum timestamp
                    pl.col('timestamp')
                    .min()
                    .alias(
                        'minimum_timestamp'
                    ),
                    # maximum timestamp
                    pl.col('timestamp')
                    .max()
                    .alias(
                        'maximum_timestamp'
                    )
                ]
            )
            .with_columns(
                (
                    # laundering rate
                    pl.col('laundering_transaction_count') / pl.col('transaction_count')
                )
                .alias(
                    'laundering_rate'
                )
            )
        )

        return self._collect(
            lazy_frame = overview_query
        )

    def get_label_distribution(self) -> pl.DataFrame:
        """ Return transaction counts and shares by target label """
        # define target label query on the transaction dataset
        label_query = (
            self.transactions
            .group_by(
                'is_laundering'
            )
            .agg(
                # count transactions per target label
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'transaction_count'
                )
            )
            .with_columns(
                [
                    # label column
                    pl.when(
                        pl.col('is_laundering') == 0
                    )
                    .then(
                        pl.lit('normal')
                    )
                    .otherwise(
                        pl.lit('laundering')
                    )
                    .alias(
                        'label'
                    ),
                    # transaction share per label
                    (
                        pl.col('transaction_count') / pl.col('transaction_count').sum()
                    )
                    .alias(
                        'transaction_share'
                    )
                ]
            )
            .select(
                [
                    'is_laundering',
                    'label',
                    'transaction_count',
                    'transaction_share'
                ]
            )
            .sort(
                'is_laundering'
            )
        )

        return self._collect(
            lazy_frame = label_query
        )

    def get_daily_activity(self) -> pl.DataFrame:
        """ Return daily transaction activity and laundering rates """
        # define transaction activity by calendar day
        daily_query = (
            self.transactions
            .with_columns(
                pl.col('timestamp')
                .dt.date()
                .alias(
                    'transaction_date'
                )
            )
            .group_by(
                'transaction_date'
            )
            .agg(
                [
                    # transaction count per calendar day
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'transaction_count'
                    ),
                    # non-legitimate transaction count per calendar day
                    pl.col('is_laundering')
                    .cast(pl.UInt64)
                    .sum()
                    .alias(
                        'laundering_transaction_count'
                    ),
                    # active unique sender count per calendar day
                    pl.col('sender_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'active_sender_count'
                    ),
                    # active unique receiver count per calendar day
                    pl.col('receiver_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'active_receiver_account'
                    )
                ]
            )
            .with_columns(
                # laundering rate per calendar day
                (
                    pl.col('laundering_transaction_count') / pl.col('transaction_count')
                )
                .alias(
                    'laundering_rate'
                )
            )
            .sort(
                'transaction_date'
            )
        )

        return self._collect(
            lazy_frame = daily_query
        )

    def get_categorical_summary(
            self,
            column_name: str
    ) -> pl.DataFrame:
        """ Return transaction volume and laundering prevalence for a categorical column """
        # check if column_name exists in categorical columns
        if column_name not in self.CATEGORICAL_COLUMNS:
            raise ValueError(
                f'Unsupported categorical column: {column_name}. '
                f'Allowed columns: {self.CATEGORICAL_COLUMNS}'
            )

        # define categorical summary query
        categorical_query = (
            self.transactions
            .group_by(column_name)
            .agg(
                [
                    # transaction count per categorical column
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'transaction_count'
                    ),
                    # non-legitimate transaction count per categorical column
                    pl.col('is_laundering')
                    .cast(pl.UInt64)
                    .sum()
                    .alias(
                        'laundering_transaction_count'
                    )
                ]
            )
            .with_columns(
                [
                    # transaction share per categorical column
                    (
                        pl.col('transaction_count') / pl.col('transaction_count').sum()
                    )
                    .alias(
                        'transaction_share'
                    ),
                    # non-legitimate transaction rate per categorical column
                    (
                        pl.col('laundering_transaction_count') / pl.col('transaction_count')
                    )
                    .alias(
                        'laundering_rate'
                    )
                ]
            )
            .sort(
                'transaction_count',
                descending = True
            )
        )

        return self._collect(
            lazy_frame = categorical_query
        )

    def get_currency_pair_summary(self) -> pl.DataFrame:
        """ Return activity by payment and received-currency pair """
        # define transaction activity by payment and received currency
        currency_query = (
            self.transactions
            .group_by(
                [
                    'payment_currency',
                    'received_currency'
                ]
            )
            .agg(
                [
                    # transaction count by payment and received currency
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'transaction_count'
                    ),
                    # non-legitimate transaction count by payment and received currency
                    pl.col('is_laundering')
                    .cast(pl.UInt64)
                    .sum()
                    .alias(
                        'laundering_transaction_count'
                    )
                ]
            )
            .with_columns(
                [
                    # same currency label
                    (
                        pl.col('payment_currency') == pl.col('received_currency')
                    )
                    .alias(
                        'same_currency'
                    ),
                    # transaction share by payment and received currency
                    (
                        pl.col('transaction_count') / pl.col('transaction_count').sum()
                    )
                    .alias(
                        'transaction_share'
                    ),
                    # non-legitimate transaction rate by payment and received currency
                    (
                        pl.col('laundering_transaction_count') / pl.col('transaction_count')
                    )
                    .alias(
                        'laundering_rate'
                    )
                ]
            )
            .sort(
                'transaction_count',
                descending = True
            )
        )

        return self._collect(
            lazy_frame = currency_query
        )

    def get_amount_summary(
            self,
            group_by: Sequence[str] = (
                'payment_currency',
                'is_laundering'
            )
    ) -> pl.DataFrame:
        """ Return amount statistics within comparable groups """
        # check if group_by contains at least one column
        group_columns = list(group_by)

        if not group_columns:
            raise ValueError(
                'group_by must contain at least one column'
            )

        # check invalid group_by columns
        invalid_columns = set(group_columns) - set(self.AMOUNT_GROUP_COLUMNS)

        if invalid_columns:
            raise ValueError(
                f'Unsupported amount grouping columns: {sorted(invalid_columns)}'
            )

        # define amount query plan by group_by columns
        amount_query = (
            self.transactions
            .group_by(
                group_columns
            )
            .agg(
                [
                    # transaction count by group_by columns
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'transaction_count'
                    ),
                    # minimum amount by group_by columns
                    pl.col('amount')
                    .min()
                    .alias(
                        'minimum_amount'
                    ),
                    # 0.25 quantile amount by group_by columns
                    pl.col('amount')
                    .quantile(
                        0.25,
                        interpolation = 'linear'
                    )
                    .alias(
                        'amount_q25'
                    ),
                    # median amount by group_by columns
                    pl.col('amount')
                    .median()
                    .alias(
                        'median_amount'
                    ),
                    # 0.75 quantile amount by group_by columns
                    pl.col('amount')
                    .quantile(
                        0.75,
                        interpolation = 'linear'
                    )
                    .alias(
                        'amount_q75'
                    ),
                    # 0.95 quantile amount by group_by columns
                    pl.col('amount')
                    .quantile(
                        0.95,
                        interpolation = 'linear'
                    )
                    .alias(
                        'amount_q95'
                    ),
                    # 0.99 quantile amount by group_by columns
                    pl.col('amount')
                    .quantile(
                        0.99,
                        interpolation = 'linear'
                    )
                    .alias(
                        'amount_q99'
                    ),
                    # maximum amount by group_by columns
                    pl.col('amount')
                    .max()
                    .alias(
                        'maximum_amount'
                    ),
                    # mean amount by group_by columns
                    pl.col('amount')
                    .mean()
                    .alias(
                        'mean_amount'
                    )
                ]
            )
            .sort(
                group_columns
            )
        )

        return self._collect(
            lazy_frame = amount_query
        )

    def get_typology_summary(
            self,
            label: int | None = None
    ) -> pl.DataFrame:
        """ 
        Return laundering_type counts 
        
        Laundering type is used for interpretation only, never as a predictive feature
        """
        # validate laundering labels
        if label not in (None, 0, 1):
            raise ValueError(
                'label must be None, 0 or 1'
            )

        # filter transactions based on label parameter
        filtered_transactions = (
            self.transactions
            if label == None
            else self.transactions.filter(
                pl.col('is_laundering') == label
            )
        )

        # define typology query
        typology_query = (
            filtered_transactions
            .group_by(
                [
                    'is_laundering',
                    'laundering_type'
                ]
            )
            .agg(
                # transaction count by typology
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'transaction_count'
                )
            )
            .with_columns(
                # transaction share within label
                (
                    pl.col('transaction_count') / pl.col('transaction_count').sum().over('is_laundering')
                )
                .alias(
                    'share_within_label'
                )
            )
            .sort(
                [
                    'is_laundering',
                    'laundering_type'
                ],
                descending = [
                    False,
                    True
                ]
            )
        )

        return self._collect(
            lazy_frame = typology_query
        )

    def build_account_profile(self) -> pl.LazyFrame:
        """  
        Build an account-level profile for identity diagnostics

        An account is provisionally suspicious if it participates in at least one laundering transaction        
        """
        return (
            self._build_account_events()
            .group_by('account')
            .agg(
                [
                    # transaction count for outgoing events by account
                    pl.col('outgoing_event')
                    .cast(pl.UInt64)
                    .sum()
                    .alias(
                        'outgoing_transaction_count'
                    ),
                    # transaction count for incoming events by account
                    pl.col('incoming_event')
                    .cast(pl.UInt64)
                    .sum()
                    .alias(
                        'incoming_transaction_count'
                    ),
                    # suspicious account label by account
                    pl.col('is_laundering')
                    .max()
                    .alias(
                        'is_suspicious_account'
                    ),
                    # bank location count by account
                    pl.col('bank_location')
                    .n_unique()
                    .cast(pl.UInt32)
                    .alias(
                        'bank_location_count'
                    )
                ]
            )
            .with_columns(
                [
                    # possible sender account
                    (
                        pl.col('outgoing_transaction_count') > 0
                    )
                    .alias(
                        'appears_as_sender'
                    ),
                    # possible receiver account
                    (
                        pl.col('incoming_transaction_count') > 0
                    )
                    .alias(
                        'appears_as_receiver'
                    )
                ]
            )
        )

    def get_account_overview(self) -> pl.DataFrame:
        """ Return account population and identity diagnostics """
        # build account-level profile
        account_profile = self.build_account_profile()

        # define account population and identity query plan
        account_query = (
            account_profile
            .select(
                [
                    # unique account count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'unique_account_count'
                    ),
                    # suspicious account count
                    pl.col('is_suspicious_account')
                    .cast(pl.UInt64)
                    .sum()
                    .alias(
                        'suspicious_account_count'
                    ),
                    # sender and receiver count
                    (
                        pl.col('appears_as_sender') & pl.col('appears_as_receiver')
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'sender_and_receiver_count'
                    ),
                    # sender only count
                    (
                        pl.col('appears_as_sender') & ~pl.col('appears_as_receiver')
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'sender_only_count'
                    ),
                    # receiver only count
                    (
                        ~pl.col('appears_as_sender') & pl.col('appears_as_receiver')
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'receiver_only_count'
                    ),
                    # multi location bank account count
                    (
                        pl.col('bank_location_count') > 1
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'multi_location_account_count'
                    )
                ]
            )
            .with_columns(
                # suspicious account rate
                (
                    pl.col('suspicious_account_count') / pl.col('unique_account_count')
                )
                .alias(
                    'suspicious_account_rate'
                )
            )
        )

        return self._collect(
            lazy_frame = account_query
        )

    def get_multi_location_accounts(
            self,
            limit: int = 20
    ) -> pl.DataFrame:
        """ Return example accounts associated with multiple locations """
        # check if limit is non-negative
        if limit <= 0:
            raise ValueError(
                'limit must be greater than 0'
            )

        # define multi location account query plan
        conflict_query = (
            self._build_account_events()
            .group_by('account')
            .agg(
                [
                    # unique bank location count
                    pl.col('bank_location')
                    .n_unique()
                    .alias(
                        'bank_location_count'
                    ),
                    # bank locations
                    pl.col('bank_location')
                    .unique()
                    .sort()
                    .alias(
                        'bank_locations'
                    ),
                    # account event count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'account_event_count'
                    )
                ]
            )
            .filter(
                pl.col('bank_location_count') > 1
            )
            .sort(
                [
                    'bank_location_count',
                    'account_event_count'
                ],
                descending = True
            )
            .head(limit)
        )

        return self._collect(
            lazy_frame = conflict_query
        )

    def get_directed_pair_summary(self) -> pl.DataFrame:
        """ Estimate the size and repetition of the future edge table """
        # define directed edge pairs
        directed_pairs = (
            self.transactions
            .group_by(
                [
                    'sender_account',
                    'receiver_account'
                ]
            )
            .agg(
                # transaction count by sender and receiver account
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'transaction_count'
                )
            )
        )

        # define pair query plan
        pair_query = (
            directed_pairs
            .select(
                [
                    # directed pair count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'directed_pair_count'
                    ),
                    # repeated directed pair count
                    (
                        pl.col('transaction_count') > 1
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'repeated_directed_pair_count'
                    ),
                    # maximum transactions per pair
                    pl.col('transaction_count')
                    .max()
                    .alias(
                        'maximum_transactions_per_pair'
                    ),
                    # mean transactions per pair
                    pl.col('transaction_count')
                    .mean()
                    .alias(
                        'mean_transactions_per_pair'
                    )
                ]
            )
        )

        return self._collect(
            lazy_frame = pair_query
        )

    def get_duplicate_summary(self) -> pl.DataFrame:
        """  
        Detect exact duplicates across all business fields

        transaction_id is deliberately excluded, since it was generated from the raw row position
        """
        # define duplicate groups
        duplicate_groups = (
            self.transactions
            .group_by(
                list(self.DUPLICATE_KEY_COLUMNS)
            )
            .agg(
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'multiplicity'
                )
            )
            .filter(
                pl.col('multiplicity') > 1
            )
        )

        # define duplicate query plan
        duplicate_query = (
            duplicate_groups
            .select(
                [
                    # duplicate group count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'duplicate_group_count'
                    ),
                    # rows count in duplicate groups
                    pl.col('multiplicity')
                    .sum()
                    .fill_null(0)
                    .alias(
                        'rows_in_duplicate_groups'
                    ),
                    # duplicate excess row count
                    (
                        pl.col('multiplicity') - 1
                    )
                    .sum()
                    .fill_null(0)
                    .alias(
                        'duplicate_excess_row_count'
                    ),
                    # maximum multiplicity
                    pl.col('multiplicity')
                    .max()
                    .fill_null(0)
                    .alias(
                        'maximum_multiplicity'
                    )
                ]
            )
        )

        return self._collect(
            lazy_frame = duplicate_query
        )