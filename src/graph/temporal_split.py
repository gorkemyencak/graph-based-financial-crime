import polars as pl

from datetime import datetime

class SAMLDTemporalSplitter:
    """
    Create leakage-safe temporal evaluation windows for graph-based financial-crime detection

    SAMLDTemporalSplitter class is responsible for temporal account construction, not for graph feaature engineering
    
    Suspicious accounts are provisionally defined as accounts that occur as the sender or receiver of a laundering transaction
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        'timestamp',
        'sender_account',
        'receiver_account',
        'is_laundering'
    )

    def __init__(
            self,
            transactions: pl.LazyFrame,
            validation_start: datetime,
            test_start: datetime,
            test_end_exclusive: datetime
    ) -> None:
        # validate chronological split boundaries
        self._validate_boundaries(
            validation_start = validation_start,
            test_start = test_start,
            test_end_exclusive = test_end_exclusive
        )

        # validate transactions schema
        self._validate_schema(
            transactions = transactions
        )

        # attributes
        self.transactions = transactions
        self.validation_start = validation_start
        self.test_start = test_start
        self.test_end_exclusive = test_end_exclusive

    ### helper private methods
    @staticmethod
    def _validate_boundaries(
        validation_start: datetime,
        test_start: datetime,
        test_end_exclusive: datetime
    ) -> None:
        """ Validate chronological split boundaries """
        if not (
            validation_start
            < test_start
            < test_end_exclusive
        ):
            raise ValueError(
                'Temporal boundaries must satisfy: '
                'validation_start < test_start < test_end_exclusive'
            )

    def _validate_schema(
            self,
            transactions: pl.LazyFrame
    ) -> None:
        """ Validate required columns """
        # get actual columns
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
                'Temporal split validation failed. '
                f'Missing columns: {sorted_missing_columns}'
            )

    @staticmethod
    def _build_suspicious_account_profile(
        transactions: pl.LazyFrame
    ) -> pl.LazyFrame:
        """ Build account profiles from laundering transactions """
        # filter laundering transactions
        suspicious_transactions = transactions.filter(
            pl.col('is_laundering') == 1
        )

        # sender related events
        sender_events = (
            suspicious_transactions
            .select(
                [
                    # account
                    pl.col('sender_account')
                    .alias(
                        'account'
                    ),
                    'timestamp',
                    # laundering sender event count
                    pl.lit(
                        1,
                        dtype = pl.UInt64
                    )
                    .alias(
                        'laundering_sender_event_count'
                    ),
                    # laundering receiver event count
                    pl.lit(
                        0,
                        dtype = pl.UInt64
                    )
                    .alias(
                        'laundering_receiver_event_count'
                    )
                ]
            )
        )

        # receiver related events
        receiver_events = (
            suspicious_transactions
            .select(
                [
                    # account
                    pl.col('receiver_account')
                    .alias(
                        'account'
                    ),
                    'timestamp',
                    # laundering sender event count
                    pl.lit(
                        0,
                        dtype = pl.UInt64
                    )
                    .alias(
                        'laundering_sender_event_count'
                    ),
                    # laundering receiver event count
                    pl.lit(
                        1,
                        dtype = pl.UInt64
                    )
                    .alias(
                        'laundering_receiver_event_count'
                    )
                ]
            )
        )

        return (
            pl.concat(
                [
                    sender_events,
                    receiver_events
                ],
                how = 'vertical'
            )
            .group_by(
                'account'
            )
            .agg(
                [
                    # first suspicious timestamp
                    pl.col('timestamp')
                    .min()
                    .alias(
                        'first_suspicious_timestamp'
                    ),
                    # last suspicious timestamp
                    pl.col('timestamp')
                    .max()
                    .alias(
                        'last_suspicious_timestamp'
                    ),
                    # total laundering sender event count
                    pl.col('laundering_sender_event_count')
                    .sum(),
                    # total laundering receiver event count
                    pl.col('laundering_receiver_event_count')
                    .sum()
                ]
            )
            .with_columns(
                # laundering account event count (sender + receiver counts)
                (
                    pl.col('laundering_sender_event_count') + pl.col('laundering_receiver_event_count')
                )
                .alias(
                    'laundering_account_event_count'
                )
            )
        )
    
    @staticmethod
    def _build_account_set(
        transactions: pl.LazyFrame
    ) -> pl.LazyFrame:
        """ Return the union of sender and receiver accounts """
        # sender accounts
        sender_accounts = (
            transactions
            .select(
                pl.col('sender_account')
                .alias(
                    'account'
                )
            )
        )

        # receiver accounts
        receiver_accounts = (
            transactions
            .select(
                pl.col('receiver_account')
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
        )

    def _build_window_discovery_summary(
            self,
            window_name: str,
            window_start: datetime,
            window_end_exclusive: datetime
    ) -> pl.LazyFrame:
        """ Build one row of account-discovery statistics """
        # future suspicious accounts
        targets = self.build_future_suspicious_accounts(
            window_start = window_start,
            window_end_exclusive = window_end_exclusive
        )

        # future suspicious account query plan
        target_counts = (
            targets
            .select(
                [
                    # future suspicious account count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'future_suspicious_account_count'
                    ),
                    # previously known future suspicious account count
                    pl.col('known_before_window')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'previously_known_future_account_count'
                    ),
                    # new future suspicious account count
                    pl.col('is_new_suspicious_account')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'new_future_suspicious_account_count'
                    ),
                    # rankable new suspicious account count
                    pl.col('is_rankable_new_suspicious_account')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'rankable_new_suspicious_account_count'
                    ),
                    # unseen new suspicious account count
                    (
                        pl.col('is_new_suspicious_account')
                        & ~pl.col('seen_before_window')
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'unseen_new_suspicious_account_count'
                    )
                ]
            )
        )

        # known suspicious account query plan
        known_counts = (
            self.build_known_suspicious_accounts(
                cutoff = window_start
            )
            .select(
                # known suspicious account count
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'known_suspicious_account_count'
                )
            )
        )

        # candidate account query plan (the union of historical sender & receiver accounts)
        candidate_counts = (
            self.build_candidate_accounts(
                cutoff = window_start
            )
            .select(
                # candidate account count
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'candidate_account_count'
                )
            )
        )

        return (
            target_counts
            .join(
                known_counts,
                how = 'cross'
            )
            .join(
                candidate_counts,
                how = 'cross'
            )
            .with_columns(
                [
                    # evaluation window
                    pl.lit(window_name)
                    .alias(
                        'evaluation_window'
                    ),
                    # window start
                    pl.lit(window_start)
                    .alias(
                        'window_start'
                    ),
                    # window end exclusive
                    pl.lit(window_end_exclusive)
                    .alias(
                        'window_end_exclusive'
                    ),
                    # rankable target coverage
                    pl.when(
                        pl.col('new_future_suspicious_account_count') > 0
                    )
                    .then(
                        pl.col('rankable_new_suspicious_account_count') / pl.col('new_future_suspicious_account_count')
                    )
                    .otherwise(
                        None
                    )
                    .alias(
                        'rankable_target_coverage'
                    ),
                    # candidate positive rate
                    pl.when(
                        pl.col('candidate_account_count') > 0
                    )
                    .then(
                        pl.col('rankable_new_suspicious_account_count') / pl.col('candidate_account_count')
                    )
                    .otherwise(
                        None
                    )
                    .alias(
                        'candidate_positive_rate'
                    )
                ]
            )
            .select(
                [
                    'evaluation_window',
                    'window_start',
                    'window_end_exclusive',
                    'known_suspicious_account_count',
                    'candidate_account_count',
                    'future_suspicious_account_count',
                    'previously_known_future_account_count',
                    'new_future_suspicious_account_count',
                    'rankable_new_suspicious_account_count',
                    'unseen_new_suspicious_account_count',
                    'rankable_target_coverage',
                    'candidate_positive_rate'
                ]
            )
        )

    ### temporal split public methods
    def assign_splits(self) -> pl.LazyFrame:
        """ Add a temporal_split column without collecting the dataset """
        return (
            self.transactions
            .with_columns(
                pl.when(
                    pl.col('timestamp') < pl.lit(self.validation_start)
                )
                .then(
                    pl.lit('history')
                )
                .when(
                    pl.col('timestamp') < pl.lit(self.test_start)
                )
                .then(
                    pl.lit('validation')
                )
                .when(
                    pl.col('timestamp') < pl.lit(self.test_end_exclusive)
                )
                .then(
                    pl.lit('test')
                )
                .otherwise(
                    pl.lit('excluded')
                )
                .alias(
                    'temporal_split'
                )
            )
        )

    def get_split_summary(self) -> pl.DataFrame:
        """ Return transaction-level statistics for each time split """
        return (
            self.assign_splits()
            .group_by(
                'temporal_split'
            )
            .agg(
                [
                    # transaction count per temporal_split
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'transaction_count'
                    ),
                    # laundering transaction count
                    pl.col('is_laundering')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'laundering_transaction_count'
                    ),
                    # unique sender count
                    pl.col('sender_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'unique_sender_count'
                    ),
                    # unique receiver count
                    pl.col('receiver_account')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'unique_receiver_count'
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
                [
                    # laundering rate
                    (
                        pl.col('laundering_transaction_count') / pl.col('transaction_count')
                    )
                    .alias(
                        'laundering_rate'
                    ),
                    # split order
                    pl.when(
                        pl.col('temporal_split') == 'history'
                    )
                    .then(0)
                    .when(
                        pl.col('temporal_split') == 'validation'
                    )
                    .then(1)
                    .when(
                        pl.col('temporal_split') == 'test'
                    )
                    .then(2)
                    .otherwise(3)
                    .alias(
                        '_split_order'
                    )
                ]
            )
            .sort(
                '_split_order'
            )
            .drop(
                '_split_order'
            )
            .collect(
                engine = 'streaming'
            )
        )

    def build_known_suspicious_accounts(
            self,
            cutoff: datetime
    ) -> pl.LazyFrame:
        """  
        Build the persosnalized PageRank seed-account table

        Only laundering transactions before cutoff are used
        """
        # filter historical transactions
        historical_transactions = (
            self.transactions
            .filter(
                pl.col('timestamp') < pl.lit(cutoff)
            )
        )

        return (
            self._build_suspicious_account_profile(
                transactions = historical_transactions
            )
            .rename(
                {
                    'first_suspicious_timestamp': 'first_known_suspicious_timestamp',
                    'last_suspicious_timestamp': 'last_known_suspicious_timestamp'
                }
            )
        )

    def build_candidate_accounts(
            self,
            cutoff: datetime
    ) -> pl.LazyFrame:
        """ 
        Return accounts that can be ranked at the cutoff 
        
        Candidate accounts:
            - appeared before the cutoff; and
            - were not previously identified as suspicious
        """
        # filter historical transactions
        historical_transactions = (
            self.transactions
            .filter(
                pl.col('timestamp') < pl.lit(cutoff)
            )
        )

        # the union of historical seneder and receiver accounts
        historical_accounts = self._build_account_set(
            transactions = historical_transactions
        )

        # build historical known suspicious accounts
        known_suspicious_accounts = (
            self.build_known_suspicious_accounts(
                cutoff = cutoff
            )
            .select('account')
        )

        return historical_accounts.join(
            known_suspicious_accounts,
            on = 'account',
            how = 'anti'
        )

    def build_future_suspicious_accounts(
            self,
            window_start: datetime,
            window_end_exclusive: datetime
    ) -> pl.LazyFrame:
        """ 
        Build target accounts for a future evaluation window

        The output distinguishes:
            - previously known suspicious accounts
            - new suspicious accounts visible in the historical graph
            - new suspicious accounts not previously seen
        """
        # sanity check on temporal windows
        if window_start >= window_end_exclusive:
            raise ValueError(
                'window_start must be earlier than window_end_exclusive.'
            )

        # filter future transactions
        future_transactions = (
            self.transactions
            .filter(
                (
                    pl.col('timestamp') >= window_start
                )
                &
                (
                    pl.col('timestamp') < window_end_exclusive
                )
            )
        )

        # future transaction suspicious accounts
        future_suspicious_accounts = (
            self._build_suspicious_account_profile(
                transactions = future_transactions
            )
            .rename(
                {
                    'first_suspicious_timestamp': 'first_future_suspicious_timestamp',
                    'last_suspicious_timestamp': 'last_future_suspicious_timestamp'
                }
            )
        )

        # historical suspicious known accounts
        known_suspicious_accounts = (
            self.build_known_suspicious_accounts(
                cutoff = window_start
            )
            .select(
                'account',
                pl.lit(True)
                .alias(
                    'known_before_window'
                )
            )
        )

        # the union of historical seneder and receiver accounts
        historical_accounts = (
            self._build_account_set(
                transactions = (
                    self.transactions
                    .filter(
                        pl.col('timestamp') < window_start
                    )
                )
            )
            .with_columns(
                pl.lit(True)
                .alias(
                    'seen_before_window'
                )
            )
        )

        return (
            future_suspicious_accounts
            .join(
                known_suspicious_accounts,
                on = 'account',
                how = 'left'
            )
            .join(
                historical_accounts,
                on = 'account',
                how = 'left'
            )
            # a missing left-join match means False
            .with_columns(
                [
                    pl.col('known_before_window')
                    .fill_null(False),
                    pl.col('seen_before_window')
                    .fill_null(False)
                ]
            )
            .with_columns(
                [
                    # new suspicious account
                    (
                        ~pl.col('known_before_window')
                    )
                    .alias(
                        'is_new_suspicious_account'
                    ),
                    # new rankable suspicious account
                    (
                        ~pl.col('known_before_window')
                        & pl.col('seen_before_window')
                    )
                    .alias(
                        'is_rankable_new_suspicious_account'
                    )
                ]
            )
        )

    def get_discovery_summary(self) -> pl.DataFrame:
        """ Summarize validation and test account-discovery targets """
        # validation summary
        validation_summary = self._build_window_discovery_summary(
            window_name = 'validation',
            window_start = self.validation_start,
            window_end_exclusive = self.test_start
        )

        # test summary
        test_summary = self._build_window_discovery_summary(
            window_name = 'test',
            window_start = self.test_start,
            window_end_exclusive = self.test_end_exclusive 
        )

        return (
            pl.concat(
                [
                    validation_summary,
                    test_summary
                ],
                how = 'vertical'
            )
            .collect(
                engine = 'streaming'
            )
        )