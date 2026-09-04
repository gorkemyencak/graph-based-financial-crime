from datetime import datetime

import polars as pl

class SAMLDGraphTableBuilder:
    """
    Build point-in-time account-node and directed-edge tables

    Graph tables contain only information available before the sanpshot cutoff. Laundering labels and typologies are
    deliberately excluded to prevent target leakage
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        'timestamp',
        'sender_account',
        'receiver_account',
        'sender_bank_location',
        'receiver_bank_location',
        'payment_type',
        'payment_currency',
        'received_currency'
    )

    def __init__(
            self,
            transactions: pl.LazyFrame
    ) -> None:
        # validate transactions schema
        self._validate_schema(
            transactions = transactions
        )

        # attributes
        self.transactions = transactions
    
    ### private helper methods
    def _validate_schema(
            self,
            transactions: pl.LazyFrame
    ) -> None:
        """ Validate the columns required for graph construction """
        # actual columns present in the transactions
        actual_columns = set(
            transactions
            .collect_schema()
            .names()
        )

        # missing columns not found in the actual columns
        missing_columns = (
            set(self.REQUIRED_COLUMNS) - actual_columns
        )

        # check if missing columns present, and return sorted missing columns
        if missing_columns:
            sorted_missing_columns = ', '.join(
                sorted(missing_columns)
            )

            raise ValueError(
                'Grap-table construction failed. '
                f'Missing columns: {sorted_missing_columns}'
            )

    @staticmethod
    def _validate_cutoff(
        cutoff: datetime
    ) -> None:
        """ Validate the graph snapshot cutoff that must be a datetime object"""
        # check if cutoff is a datetime object
        if not isinstance(
            cutoff,
            datetime
        ):
            raise TypeError(
                'cutoff must be a datetime instance.'
            )

    def _filter_history(
            self,
            cutoff: datetime
    ) -> pl.LazyFrame:
        """ Return transactions strictly before the cutoff """
        # validate cutoff provided as datetime
        self._validate_cutoff(cutoff = cutoff)

        # filter and return transactions before cutoff
        return (
            self.transactions
            .filter(
                pl.col('timestamp') < pl.lit(cutoff)
            )
        )

    @staticmethod
    def _build_account_index(
        transactions: pl.LazyFrame
    ) -> pl.LazyFrame:
        """  
        Build a deterministic account-to-node ID mapping

        Account strings are sorted before assigning node IDs so that independently constructed tables use the same mapping
        """
        # sender accounts
        sender_accounts = (
            transactions
            .select(
                pl.col('sender_accounts')
                .alias(
                    'account'
                )
            )
        )

        # receiver accounts
        receiver_accounts = (
            transactions
            .select(
                pl.col('receiver_accounts')
                .alias(
                    'account'
                )
            )
        )

        return (
            pl.concat(
                [
                    sender_accounts,
                    receiver_accounts
                ],
                how = 'vertical'
            )
            .unique(
                subset = ['account'],
                maintain_order = False
            )
            .sort(
                'account'
            )
            .with_row_index(
                name = 'node_id',
                offset = 0
            )
            .with_columns(
                pl.col('node_id')
                .cast(pl.UInt64)
            )
            .select(
                [
                    'node_id',
                    'account'
                ]
            )
        )

    ### public graph-table methods
    def build_node_table(
            self,
            cutoff: datetime
    ) -> pl.LazyFrame:
        """  
        Build the account-node table for a point-in-time snapshot
        
        An account is included if it appeared as either sender or receiver before the cutoff
        """
        # filter historical transactions
        historical_transactions = self._filter_history(
            cutoff = cutoff
        )

        # account-to-node ID mapping
        account_index = self._build_account_index(
            transactions = historical_transactions
        )

        # outgoing summary
        outgoing_summary = (
            historical_transactions
            .group_by(
                'sender_account'
            )
            .agg(
                [
                    # outgoing transaction count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'outgoing_transaction_count'
                    ),
                    # outgoing degree
                    pl.col('receiver_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'out_degree'
                    ),
                    # outgoing first seen timestamp
                    pl.col('timestamp')
                    .min()
                    .alias(
                        'outgoing_first_seen_timestamp'
                    ),
                    # outgoing last seen timestamp
                    pl.col('timestamp')
                    .max()
                    .alias(
                        'outgoing_last_seen_timestamp'
                    )
                ]
            )
            .rename(
                {
                    'sender_account': 'account'
                }
            )
        )

        # incoming summary
        incoming_summary = (
            historical_transactions
            .group_by(
                'receiver_account'
            )
            .agg(
                [
                    # incoming transaction count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'incoming_transaction_count'
                    ),
                    # incoming degree
                    pl.col('sender_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'in_degree'
                    ),
                    # incoming first seen timestamp
                    pl.col('timestamp')
                    .min()
                    .alias(
                        'incoming_first_seen_timestamp'
                    ),
                    # incoming last seen timestamp
                    pl.col('timestamp')
                    .max()
                    .alias(
                        'incoming_last_seen_timestamp'
                    )
                ]
            )
            .rename(
                {
                    'receiver_account': 'account'
                }
            )
        )

        # sender locations
        sender_locations = (
            historical_transactions
            .select(
                [
                    # sender account
                    pl.col('sender_account')
                    .alias(
                        'account'
                    ),
                    # bank location
                    pl.col('sender_bank_location')
                    .alias(
                        'bank_location'
                    )
                ]
            )
        )

        # receiver locations
        receiver_locations = (
            historical_transactions
            .select(
                [
                    # receiver account
                    pl.col('receiver_account')
                    .alias(
                        'account'
                    ),
                    # bank location
                    pl.col('receiver_bank_location')
                    .alias(
                        'bank_location'
                    )
                ]
            )
        )

        # location summary
        location_summary = (
            pl.concat(
                [
                    sender_locations,
                    receiver_locations
                ],
                how = 'vertical'
            )
            .drop_nulls(
                subset = [
                    'account',
                    'bank_location'
                ]
            )
            .unique(
                subset = [
                    'account',
                    'bank_location'
                ],
                maintain_order = False
            )
            .group_by(
                'account'
            )
            .agg(
                # bank location count per account
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'bank_location_count'
                )
            )
        )

        # define count columns
        count_columns = [
            'outgoing_transaction_count',
            'incoming_transaction_count',
            'out_degree',
            'in_degree',
            'bank_location_count'
        ]

        return (
            account_index
            .join(
                outgoing_summary,
                on = 'account',
                how = 'left'
            )
            .join(
                incoming_summary,
                on = 'account',
                how = 'left'
            )
            .join(
                location_summary,
                on = 'account',
                how = 'left'
            )
            .with_columns(
                [
                    pl.col(column_name)
                    .fill_null(0)
                    .cast(pl.UInt64)
                    for column_name in count_columns
                ]
            )
            .with_columns(
                [
                    # first seen timestamp
                    pl.when(
                        pl.col('outgoing_first_seen_timestamp')
                        .is_null()
                    )
                    .then(
                        pl.col('incoming_first_seen_timestamp')
                    )
                    .when(
                        pl.col('incoming_first_seen_timestamp')
                        .is_null()
                    )
                    .then(
                        pl.col('outgoing_first_seen_timestamp')
                    )
                    .otherwise(
                        pl.min_horizontal(
                            'outgoing_first_seen_timestamp',
                            'incoming_first_seen_timestamp'
                        )
                    )
                    .alias(
                        'first_seen_timestamp'
                    ),
                    # last seen timestamp
                    pl.when(
                        pl.col('outgoing_last_seen_timestamp')
                        .is_null()
                    )
                    .then(
                        pl.col('incoming_last_seen_timestamp')
                    )
                    .when(
                        pl.col('incoming_last_seen_timestamp')
                        .is_null()
                    )
                    .then(
                        pl.col('outgoing_last_seen_timestamp')
                    )
                    .otherwise(
                        pl.max_horizontal(
                            'outgoing_last_seen_timestamp',
                            'incoming_last_seen_timestamp'
                        )
                    )
                    .alias(
                        'last_seen_timestamp'
                    ),
                    # account transaction event count
                    (
                        pl.col('outgoing_transaction_count') + pl.col('incoming_transaction_count')
                    )
                    .alias(
                        'account_transaction_event_count'
                    ),
                    # total directed degree
                    (
                        pl.col('in_degree') + pl.col('out_degree')
                    )
                    .alias(
                        'total_directed_degree'
                    ),
                    # outgoing activity label
                    (
                        pl.col('outgoing_transaction_count') > 0
                    )
                    .alias(
                        'has_outgoing_activity'
                    ),
                    # incoming activity label
                    (
                        pl.col('incoming_transaction_count') > 0
                    )
                    .alias(
                        'has_incoming_activity'
                    ),
                    # multi location label
                    (
                        pl.col('bank_location_count') > 1
                    )
                    .alias(
                        'is_multi_location'
                    )
                ]
            )
            .drop(
                [
                    'outgoing_first_seen_timestamp',
                    'outgoing_last_seen_timestamp',
                    'incoming_first_seen_timestamp',
                    'incoming_last_seen_timestamp'
                ]
            )
            .select(
                [
                    'node_id',
                    'account',
                    'first_seen_timestamp',
                    'last_seen_timestamp',
                    'outgoing_transaction_count',
                    'incoming_transaction_count',
                    'account_transaction_event_count',
                    'out_degree',
                    'in_degree',
                    'total_directed_degree',
                    'bank_location_count',
                    'is_multi_location',
                    'has_outgoing_activity',
                    'has_incoming_activity'
                ]
            )
        )