import json

from datetime import datetime
from pathlib import Path

import polars as pl
import igraph as ig

from src.graph.builder import SAMLDGraphTableBuilder

class RingExperimentPaths:
    """ Paths belonging to one persisted validation ring experiment """
    def __init__(
            self,
            directory: Path | str,
            nodes: Path | str,
            edges: Path | str,
            rings: Path | str,
            ring_accounts: Path | str,
            seeds: Path | str,
            targets: Path | str,
            manifest: Path | str
    ) -> None:
        # attributes
        self.directory = Path(directory)
        self.nodes = Path(nodes)
        self.edges = Path(edges)
        self.rings = Path(rings)
        self.ring_accounts = Path(ring_accounts)
        self.seeds = Path(seeds)
        self.targets = Path(targets)
        self.manifest = Path(manifest)

    def parquet_paths(self) -> dict[str, Path]:
        """ Return the experiment table names and their Parquet paths """
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'rings': self.rings,
            'ring_accounts': self.ring_accounts,
            'seeds': self.seeds,
            'targets': self.targets
        }

    def as_dict(self) -> dict[str, Path]:
        """ Return all experiment artifact graphs """
        return {
            **self.parquet_paths(),
            'manifest': self.manifest
        }


class SAMLDRingExperimentBuilder:
    """
    Prepare a validation-only, seed-withholding ring-expansion experiment

    SAML-D does not contain a native ring identifier. This class therefore defines an approximate ring as a weakly connected
    component of the suspicious-transaction subgraph inside the validation window. Components smaller than 'minimum_ring_size'
    are excluded

    Within each eligible ring, accounts are ordered by their first suspicious appearance. The earliest fraction becomes the known seed set
    and the remaining accounts become held-out ring-expansion targets

    The observation graph contains all transactions strictly before 'validation_end_exclusive'. Laundering labels and typologies are used only 
    to construct retrospective evaluation tables; they are never added to the observation node or edge tables
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        'timestamp',
        'sender_account',
        'receiver_account',
        'payment_currency',
        'received_currency',
        'sender_bank_location',
        'receiver_bank_location',
        'payment_type',
        'is_laundering',
        'laundering_type'
    )

    _REQUIRED_OUTPUT_COLUMNS: dict[str, tuple[str, ...]] = {
        'nodes': (
            'node_id',
            'account'
        ),
        'edges': (
            'source_node_id',
            'target_node_id',
            'count_weight',
            'log_count_weight'
        ),
        'rings': (
            'ring_id',
            'ring_size',
            'seed_count',
            'target_count'
        ),
        'ring_accounts': (
            'ring_id',
            'node_id',
            'account',
            'is_ring_seed',
            'is_ring_target'
        ),
        'seeds': (
            'ring_id',
            'node_id',
            'account',
            'seed_order'
        ),
        'targets': (
            'ring_id',
            'node_id',
            'account',
            'target_order'
        )
    }

    def __init__(
            self,
            transactions: pl.LazyFrame,
            validation_start: datetime,
            validation_end_exclusive: datetime,
            output_dir: Path | str,
            minimum_ring_size: int = 3,
            seed_fraction: float | int = 0.25,
            minimum_seed_count: int = 1
    ) -> None:
        # validate transactions schema
        self._validate_schema(
            transactions = transactions
        )

        # validate temporal split window
        self._validate_window(
            validation_start = validation_start,
            validation_end_exclusive = validation_end_exclusive
        )

        # validate experiment parameters
        self._validate_experiment_parameters(
            minimum_ring_size = minimum_ring_size,
            seed_fraction = seed_fraction,
            minimum_seed_count = minimum_seed_count
        )
        
        # attributes
        self.transactions = transactions
        self.validation_start = validation_start
        self.validation_end_exclusive = validation_end_exclusive
        self.output_dir = output_dir
        self.minimum_ring_size = minimum_ring_size
        self.seed_fraction = seed_fraction
        self.minimum_seed_count = minimum_seed_count

        self.graph_builder = SAMLDGraphTableBuilder(
            transactions = self.transactions
        )

        # class-level cache
        self._validation_transactions_cache: pl.DataFrame | None = None
        self._component_summary_cache: pl.DataFrame | None = None
        self._ring_tables_cache: dict[str, pl.DataFrame] | None = None

    ### private validation methods
    @classmethod
    def _validate_schema(
        cls,
        transactions: pl.LazyFrame
    ) -> None:
        """ Validate the transaction columns required by the experiment """
        # validate whether transactions is a pl.LazyFrame instance
        if not isinstance(transactions, pl.LazyFrame):
            raise TypeError(
                'transactions must be a Polars LazyFrame'
            )

        # extract actual column names from transactions
        actual_columns = set(
            transactions
            .collect_schema()
            .names()
        )

        # expected columns
        expected_columns = set(
            cls.REQUIRED_COLUMNS
        )

        # missing columns
        missing_columns = expected_columns - actual_columns

        # check if missing_columns, and return sorted missing columns if present
        if missing_columns:
            sorted_missing_columns = ', '.join(
                sorted(missing_columns)
            )

            raise ValueError(
                f'Ring-experiment preparation failed. Missing columns: {sorted_missing_columns}'
            )

    @staticmethod
    def _validate_window(
        validation_start: datetime,
        validation_end_exclusive: datetime
    ) -> None:
        """ Validate the half-open validation window """
        # validate whether validation_start is a datetime instance
        if not isinstance(validation_start, datetime):
            raise TypeError(
                'validation_start must be a datetime instance'
            )

        # validate whether validation_end_exclusive is a datetime instance
        if not isinstance(validation_end_exclusive, datetime):
            raise TypeError (
                'validation_end_exclusive must be a datetime instance'
            )

        # logical expreassion
        if validation_start >= validation_end_exclusive:
            raise ValueError(
                'validation_start must be earlier than validation_end_exclusive'
            )

    @staticmethod
    def _validate_experiment_parameters(
        minimum_ring_size: int,
        seed_fraction: float | int,
        minimum_seed_count: int
    ) -> None:
        """ Validate component eligibility and seed-withholding settings """
        # validate instance types
        if not isinstance(minimum_ring_size, int):
            raise TypeError(
                'minimum_ring_size must be an integer'
            )

        if not isinstance(seed_fraction, int | float):
            raise TypeError(
                'seed_fractions must be either int or float instance'
            )

        if not isinstance(minimum_seed_count, int):
            raise TypeError(
                'minimum_seed_count must be an int instance'
            )

        # validate class-level limitations & logical expressions
        if minimum_ring_size < 2:
            raise ValueError(
                'minimum_ring_size must be at least 2'
            )

        if not 0.0 < float(seed_fraction) < 1.0:
            raise ValueError(
                'seed_fraction must be strictly between 0.0 and 1.0'
            )

        if minimum_seed_count < 1:
            raise ValueError(
                'minimum_seed_count must be at least 1'
            )

        if minimum_seed_count >= minimum_ring_size:
            raise ValueError(
                'minimum_seed_count must be smaller than '
                'minimum_ring_size so every ring has a target'
            )

    @staticmethod
    def _get_temporary_path(
        path: Path
    ) -> Path:
        """ Return a temporary sibling path for an experiment artifact """
        return path.with_name(
            name = f'.{path.stem}.tmp{path.suffix}'
        )

    @staticmethod
    def _sink_parquet(
        lazy_frame: pl.LazyFrame,
        path: Path
    ) -> None:
        """ Stream a lazy table to compressed Parquet """
        lazy_frame.sink_parquet(
            path = path,
            compression = 'zstd',
            statistics = True,
            maintain_order = True,
            engine = 'streaming'
        )

    @classmethod
    def _validate_parquet_file(
        cls,
        table_name: str,
        path: Path
    ) -> None:
        """ Validate one persisted experiment table """
        # validate if path is a file
        if not path.is_file():
            raise FileNotFoundError(
                f'Ring-experiment table does not exist: {path}'
            )

        # validate if path is empty on disk
        if path.stat().st_size == 0:
            raise ValueError(
                f'Ring-experiment table is empty: {path}'
            )

        # extract actual columns from Parquet schema
        actual_columns = set(
            pl.scan_parquet(source = path)
            .collect_schema()
            .names()
        )

        # expected columns
        expected_columns = set(
            cls._REQUIRED_OUTPUT_COLUMNS[table_name]
        )

        # missing columns
        missing_columns = expected_columns - actual_columns

        # check if missing_columns, and return sorted missing columns if present
        if missing_columns:
            sorted_missing_columns = ', '.join(
                sorted(missing_columns)
            )

            raise ValueError(
                f'{table_name!r} ring-experiment table has missing columns: {sorted_missing_columns}'
            )

    @staticmethod
    def _multi_source_reachable_nodes(
        graph: ig.Graph,
        seed_node_ids: list[int],
        direction: str
    ) -> set[int]:
        """ Return nodes reachable from pooled seeds in one direction """
        # create a graph copy
        graph_copy = graph.copy()
        super_source_node_id = graph_copy.vcount()
        graph_copy.add_vertex()

        # add edges to graph depending on direction, and compute reachable nodes
        if direction == 'forward':
            graph_copy.add_edges(
                [
                    (super_source_node_id, seed_node_id)
                    for seed_node_id in seed_node_ids
                ]
            )

            reachable_nodes = graph_copy.subcomponent(
                super_source_node_id,
                mode = 'out'
            )
        elif direction == 'reverse':
            graph_copy.add_edges(
                [
                    (seed_node_id, super_source_node_id)
                    for seed_node_id in seed_node_ids
                ]
            )

            reachable_nodes = graph_copy.subcomponent(
                super_source_node_id,
                mode = 'in'
            )
        elif direction == 'weak':
            graph_copy.to_undirected(
                mode = 'collapse'
            )
            graph_copy.add_edges(
                [
                    (super_source_node_id, seed_node_id)
                    for seed_node_id in seed_node_ids
                ]
            )

            reachable_nodes = graph_copy.subcomponent(
                super_source_node_id,
                mode = 'all'
            )
        else:
            raise ValueError(
                "direction must be 'forward', 'reverse', or 'weak'"
            )

        return {
            int(node_id)
            for node_id in reachable_nodes
            if int(node_id) != super_source_node_id
        }

    ### private ring-construction methods
    def _collect_validation_suspicious_transactions(self) -> pl.DataFrame:
        """ Collect labeled transactions inside the validation window """
        # build validation transactions, if self._validation_transactions_cache is empty
        if self._validation_transactions_cache is None:
            # build validation transactions
            validation_transactions = (
                self.transactions
                .filter(
                    (pl.col('timestamp') >= pl.lit(self.validation_start))
                    & (pl.col('timestamp') < pl.lit(self.validation_end_exclusive))
                    & (pl.col('is_laundering') == 1)
                )
                .select(
                    [
                        'timestamp',
                        'sender_account',
                        'receiver_account',
                        'payment_currency',
                        'received_currency',
                        'payment_type',
                        'laundering_type'
                    ]
                )
                .sort(
                    [
                        'timestamp',
                        'sender_account',
                        'receiver_account'
                    ]
                )
                .collect(
                    engine = 'streaming'
                )
            )

            if validation_transactions.is_empty():
                raise ValueError(
                    'The validation window contain no suspicious transactions'
                )

            self._validation_transactions_cache = validation_transactions

        return self._validation_transactions_cache.clone()

    def _build_observation_node_index(self) -> pl.LazyFrame:
        """ Build the deterministic account index used by the observation graph """
        # observed transactions before the validation end excluvise period
        observed_transactions = (
            self.transactions
            .filter(
                pl.col('timestamp') < pl.lit(self.validation_end_exclusive)
            )
        )

        return (
            pl.concat(
                [
                    observed_transactions
                    .select(
                        pl.col('sender_account')
                        .alias(
                            'account'
                        )
                    ),
                    observed_transactions
                    .select(
                        pl.col('receiver_account')
                        .alias(
                            'account'
                        )
                    )
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

    def _build_component_membership(
            self,
            suspicious_transactions: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """ Build deterministic weak-component membership for accounts """
        # account-index
        account_index = (
            pl.concat(
                [
                    suspicious_transactions
                    .select(
                        pl.col('sender_account')
                        .alias(
                            'account'
                        )
                    ),
                    suspicious_transactions
                    .select(
                        pl.col('receiver_account')
                        .alias(
                            'account'
                        )
                    )
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
                name = 'component_vertex_id',
                offset = 0
            )
            .with_columns(
                pl.col('component_vertex_id')
                .cast(pl.UInt64)
            )
        )

        # component edges
        component_edges = (
            suspicious_transactions
            .select(
                [
                    'sender_account',
                    'receiver_account'
                ]
            )
            .unique(
                maintain_order = False
            )
            .join(
                account_index
                .select(
                    [
                        # sender account
                        pl.col('account')
                        .alias(
                            'sender_account'
                        ),
                        # source vertex id
                        pl.col('component_vertex_id')
                        .alias(
                            'source_vertex_id'
                        )
                    ]
                ),
                on = 'sender_account',
                how = 'left'
            )
            .join(
                account_index
                .select(
                    [
                        # receiver account
                        pl.col('account')
                        .alias(
                            'receiver_account'
                        ),
                        # target vertex id
                        pl.col('component_vertex_id')
                        .alias(
                            'target_vertex_id'
                        )
                    ]
                ),
                on = 'receiver_account',
                how = 'left'
            )
        )

        # igraph
        graph = ig.Graph(
            n = account_index.height,
            edges = [
                (int(source_node_id), int(target_node_id))
                for source_node_id, target_node_id in (
                    component_edges
                    .select(
                        [
                            'source_vertex_id',
                            'target_vertex_id'
                        ]
                    )
                    .iter_rows()
                )
            ],
            directed = True
        )

        # raw membership
        raw_membership = (
            account_index
            .with_columns(
                pl.Series(
                    name = 'raw_component_id',
                    values = graph.connected_components(
                        mode = 'weak'
                    )
                    .membership,
                    dtype = pl.UInt64
                )
            )
            .select(
                [
                    'account',
                    'raw_component_id'
                ]
            )
        )

        # component summary
        component_summary = (
            raw_membership
            .group_by(
                'raw_component_id'
            )
            .agg(
                [
                    # component size
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'component_size'
                    ),
                    # canonical account
                    pl.col('account')
                    .min()
                    .alias(
                        'canonical_account'
                    )
                ]
            )
            .with_columns(
                (
                    pl.col('component_size') >= self.minimum_ring_size
                )
                .alias(
                    'is_eligible_ring'
                )
            )
            .sort(
                [
                    'canonical_account',
                    'raw_component_id'
                ]
            )
        )

        # eligible components
        eligible_components = (
            component_summary
            .filter(
                pl.col('is_eligible_ring')
            )
            .with_row_index(
                name = 'ring_id',
                offset = 0
            )
            .with_columns(
                pl.col('ring_id')
                .cast(pl.UInt64)
            )
            .select(
                [
                    'raw_component_id',
                    'ring_id'
                ]
            )
        )

        # join eligible components to component summary
        component_summary = (
            component_summary
            .join(
                eligible_components,
                on = 'raw_component_id',
                how = 'left'
            )
            .select(
                [
                    'raw_component_id',
                    'ring_id',
                    'canonical_account',
                    'component_size',
                    'is_eligible_ring'
                ]
            )
            .sort(
                by = [
                    'is_eligible_ring',
                    'component_size',
                    'canonical_account'
                ],
                descending = [
                    True,
                    True,
                    False
                ]
            )
        )

        # eligible membership
        eligible_membership = (
            raw_membership
            .join(
                eligible_components,
                on = 'raw_component_id',
                how = 'inner'
            )
            .select(
                [
                    'ring_id',
                    'account'
                ]
            )
            .sort(
                [
                    'ring_id',
                    'account'
                ]
            )
        )

        # validate if eligible membership is empty
        if eligible_membership.is_empty():
            raise ValueError(
                'No suspicious validation component satisfies '
                f'minimum_ring_size={self.minimum_ring_size}'
            )

        return eligible_membership, component_summary

    @staticmethod
    def _build_account_activity(
        suspicious_transactions: pl.DataFrame
    ) -> pl.DataFrame:
        """ Summarize suspicious validation activity for each account """
        # sender account activity
        sender_activity = (
            suspicious_transactions
            .group_by(
                'sender_account'
            )
            .agg(
                [
                    # sender first suspicious timestamp
                    pl.col('timestamp')
                    .min()
                    .alias(
                        'sender_first_suspicious_timestamp'
                    ),
                    # sender last suspicious timestamp
                    pl.col('timestamp')
                    .max()
                    .alias(
                        'sender_last_suspicious_timestamp'
                    ),
                    # suspicious sender event count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'suspicious_sender_event_count'
                    )
                ]
            )
            .rename(
                {
                    'sender_account': 'account'
                }
            )
        )

        # receiver account activity
        receiver_activity = (
            suspicious_transactions
            .group_by(
                'receiver_account'
            )
            .agg(
                [
                    # receiver first suspicious timestamp
                    pl.col('timestamp')
                    .min()
                    .alias(
                        'receiver_first_suspicious_timestamp'
                    ),
                    # receiver last suspicious timestamp
                    pl.col('timestamp')
                    .max()
                    .alias(
                        'receiver_last_suspicious_timestamp'
                    ),
                    # suspicious receiver event count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'suspicious_receiver_event_count'
                    )
                ]
            )
            .rename(
                {
                    'receiver_account': 'account'
                }
            )
        )

        # all accounts
        all_accounts = (
            pl.concat(
                [
                    sender_activity
                    .select(
                        'account'
                    ),
                    receiver_activity
                    .select(
                        'account'
                    )
                ],
                how = 'vertical'
            )
            .unique(
                maintain_order = False
            )
        )

        return (
            all_accounts
            .join(
                sender_activity,
                on = 'account',
                how = 'left'
            )
            .join(
                receiver_activity,
                on = 'account',
                how = 'left'
            )
            .with_columns(
                [
                    # first suspicious timestamp
                    pl.min_horizontal(
                        'sender_first_suspicious_timestamp',
                        'receiver_first_suspicious_timestamp'
                    )
                    .alias(
                        'first_suspicious_timestamp'
                    ),
                    # last suspicious timestamp
                    pl.max_horizontal(
                        'sender_last_suspicious_timestamp',
                        'receiver_last_suspicious_timestamp'
                    )
                    .alias(
                        'last_suspicious_timestamp'
                    ),
                    # suspicious sender event count
                    pl.col('suspicious_sender_event_count')
                    .fill_null(0)
                    .cast(pl.UInt64),
                    # suspicious receiver event count
                    pl.col('suspicious_receiver_event_count')
                    .fill_null(0)
                    .cast(pl.UInt64)
                ]
            )
            .with_columns(
                (
                    pl.col('suspicious_sender_event_count') + pl.col('suspicious_receiver_event_count')
                )
                .cast(pl.UInt64)
                .alias(
                    'suspicious_account_event_count'
                )
            )
            .select(
                [
                    'account',
                    'first_suspicious_timestamp',
                    'last_suspicious_timestamp',
                    'suspicious_sender_event_count',
                    'suspicious_receiver_event_count',
                    'suspicious_account_event_count'
                ]
            )
        )

    def _build_ring_summary(
            self,
            suspicious_transactions: pl.DataFrame,
            ring_accounts: pl.DataFrame
    ) -> pl.DataFrame:
        """ Summarize the transactions and withheld labels of each ring """
        # ring index
        ring_index = (
            ring_accounts
            .select(
                [
                    'ring_id',
                    'account'
                ]
            )
        )

        # ring transactions
        ring_transactions = (
            suspicious_transactions
            .join(
                ring_index
                .rename(
                    {
                        'ring_id': 'sender_ring_id',
                        'account': 'sender_account'
                    }
                ),
                on = 'sender_account',
                how = 'inner'
            )
            .join(
                ring_index
                .rename(
                    {
                        'ring_id': 'receiver_ring_id',
                        'account': 'receiver_account'
                    }
                ),
                on = 'receiver_account',
                how = 'inner'
            )
        )

        # mismatched ring edges
        mismatched_ring_edges = (
            ring_transactions
            .filter(
                pl.col('sender_ring_id') != pl.col('receiver_ring_id')
            )
            .height
        )

        if mismatched_ring_edges > 0:
            raise ValueError(
                'A suspicious validation edge connects two different ring identifiers'
            )

        # transaction summary
        transaction_summary = (
            ring_transactions
            .group_by(
                'sender_ring_id'
            )
            .agg(
                [
                    # suspicious transaction count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'suspicious_transaction_count'
                    ),
                    # suspicious directed edge count
                    pl.struct(
                        [
                            'sender_account',
                            'receiver_account'
                        ]
                    )
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'suspicious_directed_edge_count'
                    ),
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
                    # suspicious active day count
                    pl.col('timestamp')
                    .dt.date()
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'suspicious_active_day_count'
                    ),
                    # laundering type count
                    pl.col('laundering_type')
                    .drop_nulls()
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'laundering_type_count'
                    ),
                    # laundering types
                    pl.col('laundering_type')
                    .drop_nulls()
                    .unique()
                    .sort()
                    .alias(
                        'laundering_types'
                    ),
                    # payment type count
                    pl.col('payment_type')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'payment_type_count'
                    ),
                    # payment currency count
                    pl.col('payment_currency')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'payment_currency_count'
                    ),
                    # received currency count
                    pl.col('received_currency')
                    .n_unique()
                    .cast(pl.UInt64)
                    .alias(
                        'received_currency_count'
                    ),
                    # cross currency transaction count
                    (
                        pl.col('payment_currency') != pl.col('received_currency')
                    )
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'cross_currency_transaction_count'
                    )
                ]
            )
            .rename(
                {
                    'sender_ring_id': 'ring_id'
                }
            )
        )

        # account summary
        account_summary = (
            ring_accounts
            .group_by(
                'ring_id'
            )
            .agg(
                [
                    # ring size
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'ring_size'
                    ),
                    # canonical account
                    pl.col('account')
                    .min()
                    .alias(
                        'canonical_account'
                    ),
                    # seed count
                    pl.col('is_ring_seed')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'seed_count'
                    ),
                    # target count
                    pl.col('is_ring_target')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'target_count'
                    )
                ]
            )
        )

        return (
            account_summary
            .join(
                transaction_summary,
                on = 'ring_id',
                how = 'left'
            )
            .with_columns(
                (
                    pl.col('laundering_type_count') > 1
                )
                .alias(
                    'is_mixed_typology_ring'
                )
            )
            .select(
                [
                    'ring_id',
                    'canonical_account',
                    'ring_size',
                    'seed_count',
                    'target_count',
                    'suspicious_transaction_count',
                    'suspicious_directed_edge_count',
                    'first_suspicious_timestamp',
                    'last_suspicious_timestamp',
                    'suspicious_active_day_count',
                    'laundering_type_count',
                    'laundering_types',
                    'is_mixed_typology_ring',
                    'payment_type_count',
                    'payment_currency_count',
                    'received_currency_count',
                    'cross_currency_transaction_count'
                ]
            )
            .sort(
                'ring_id'
            )
        )

    ### public construction methods
    def build_observation_graph_tables(self) -> dict[str, pl.LazyFrame]:
        """ Build label-free graph tables through the validation window """
        return {
            'nodes': self.graph_builder.build_node_table(
                cutoff = self.validation_end_exclusive
            ),
            'edges': self.graph_builder.build_edge_table(
                cutoff = self.validation_end_exclusive
            )
        }

    def build_component_summary(
            self,
            force_rebuild: bool = False
    ) -> pl.DataFrame:
        """ Return all suspicious component sizes and eligibility flags """
        # validate if self._component_summary_cache is empty or force_rebuild is True
        if self._component_summary_cache is None or force_rebuild:
            # suspicious transactions
            suspicious_transactions = self._collect_validation_suspicious_transactions()

            # component membership
            _, component_summary = self._build_component_membership(
                suspicious_transactions = suspicious_transactions
            )

            self._component_summary_cache = component_summary

        return self._component_summary_cache

    def build_ring_tables(
            self,
            force_rebuild: bool = False
    ) -> dict[str, pl.DataFrame]:
        """ Build ring, account, seed, and withheld-target variables """
        # validate if self._ring_table_cache is empty or force_rebuild is True
        if self._ring_tables_cache is None or force_rebuild:
            # suspicious transactions
            suspicious_transactions = self._collect_validation_suspicious_transactions()

            # component membership
            component_membership, component_summary = self._build_component_membership(
                suspicious_transactions = suspicious_transactions
            )

            self._component_summary_cache = component_summary

            # account activity
            account_activity = self._build_account_activity(
                suspicious_transactions = suspicious_transactions
            )

            # ring accounts
            ring_accounts = (
                component_membership
                .join(
                    account_activity,
                    on = 'account',
                    how = 'left'
                )
            )

            # ring sizes
            ring_sizes = (
                ring_accounts
                .group_by(
                    'ring_id'
                )
                .agg(
                    # ring size
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'ring_size'
                    )
                )
                .with_columns(
                    # requested seed count
                    pl.max_horizontal(
                        [
                            # minimum seed count
                            pl.lit(
                                self.minimum_seed_count,
                                dtype = pl.UInt64
                            ),
                            # ring_size * self.seed_fraction
                            (
                                pl.col('ring_size')
                                .cast(pl.Float64)
                                * self.seed_fraction
                            )
                            .ceil()
                            .cast(pl.UInt64)
                        ]
                    )
                    .alias(
                        'requested_seed_count'
                    )
                )
                .with_columns(
                    # seed count
                    pl.min_horizontal(
                        pl.col('requested_seed_count'),
                        pl.col('ring_size') - 1
                    )
                    .cast(pl.UInt64)
                    .alias(
                        'seed_count'
                    )
                )
                .with_columns(
                    # target count
                    (
                        pl.col('ring_size') - pl.col('seed_count')
                    )
                    .cast(pl.UInt64)
                    .alias(
                        'target_count'
                    )
                )
                .drop(
                    'requested_seed_count'
                )
            )

            # ring accounts with ring sizes
            ring_accounts = (
                ring_accounts
                .join(
                    ring_sizes,
                    on = 'ring_id',
                    how = 'left'
                )
                .sort(
                    [
                        'ring_id',
                        'first_suspicious_timestamp',
                        'account'
                    ]
                )
                .with_columns(
                    # ring member order
                    pl.col('account')
                    .cum_count()
                    .over('ring_id')
                    .cast(pl.UInt64)
                    .alias(
                        'ring_member_order'
                    )
                )
                .with_columns(
                    [
                        # ring seed label
                        (
                            pl.col('ring_member_order') <= pl.col('seed_count')
                        )
                        .alias(
                            'is_ring_seed'
                        ),
                        # ring target label
                        (
                            pl.col('ring_member_order') > pl.col('seed_count')
                        )
                        .alias(
                            'is_ring_target'
                        )
                    ]
                )
            )

            # node index
            node_index = self._build_observation_node_index()

            # ring accounts with node index
            ring_accounts = (
                ring_accounts
                .lazy()
                .join(
                    node_index,
                    on = 'account',
                    how = 'left'
                )
                .collect(
                    engine = 'streaming'
                )
            )

            # missing node count, and validate those missing nodes
            missing_node_count = (
                ring_accounts
                .get_column('node_id')
                .null_count()
            )

            if missing_node_count > 0:
                raise ValueError(
                    f'{missing_node_count:,} ring accounts do not map to the validation observation graph'
                )

            ring_accounts = (
                ring_accounts
                .select(
                    [
                        'ring_id',
                        'node_id',
                        'account',
                        'ring_member_order',
                        'ring_size',
                        'seed_count',
                        'target_count',
                        'first_suspicious_timestamp',
                        'last_suspicious_timestamp',
                        'suspicious_sender_event_count',
                        'suspicious_receiver_event_count',
                        'suspicious_account_event_count',
                        'is_ring_seed',
                        'is_ring_target'
                    ]
                )
                .sort(
                    [
                        'ring_id',
                        'ring_member_order'
                    ]
                )
            )

            # ring summary
            rings = self._build_ring_summary(
                suspicious_transactions = suspicious_transactions,
                ring_accounts = ring_accounts
            )

            # seeds
            seeds = (
                ring_accounts
                .filter(
                    pl.col('is_ring_seed')
                )
                .with_columns(
                    # seed order
                    pl.col('ring_member_order')
                    .alias(
                        'seed_order'
                    )
                )
                .select(
                    [
                        'ring_id',
                        'node_id',
                        'account',
                        'seed_order',
                        'ring_size',
                        'seed_count',
                        'target_count',
                        'first_suspicious_timestamp',
                        'last_suspicious_timestamp',
                        'suspicious_sender_event_count',
                        'suspicious_receiver_event_count',
                        'suspicious_account_event_count'
                    ]
                )
                .sort(
                    [
                        'ring_id',
                        'seed_order'
                    ]
                )
            )

            # targets
            targets = (
                ring_accounts
                .filter(
                    pl.col('is_ring_target')
                )
                .with_columns(
                    # target order
                    (
                        pl.col('ring_member_order') - pl.col('seed_count')
                    )
                    .cast(pl.UInt64)
                    .alias(
                        'target_order'
                    )
                )
                .select(
                    [
                        'ring_id',
                        'node_id',
                        'account',
                        'target_order',
                        'ring_size',
                        'seed_count',
                        'target_count',
                        'first_suspicious_timestamp',
                        'last_suspicious_timestamp',
                        'suspicious_sender_event_count',
                        'suspicious_receiver_event_count',
                        'suspicious_account_event_count'
                    ]
                )
                .sort(
                    [
                        'ring_id',
                        'target_order'
                    ]
                )
            )

            # populate self._ring_tables_cache
            self._ring_tables_cache = {
                'rings': rings,
                'ring_accounts': ring_accounts,
                'seeds': seeds,
                'targets': targets
            }

        return {
            table_name: table.clone()
            for table_name, table in self._ring_tables_cache.items()
        }

    def build_experiment_summary(self) -> pl.DataFrame:
        """ Return one-word preparation statistics for the experiment """
        # suspicious transactions
        suspicious_transactions = self._collect_validation_suspicious_transactions()

        # component summary
        component_summary = self.build_component_summary(
            force_rebuild = False
        )

        # ring tables
        ring_tables = self.build_ring_tables(
            force_rebuild = False
        )

        # rings
        rings = ring_tables['rings']

        return pl.DataFrame(
            {
                'validation_start': [self.validation_start],
                'validation_end_exclusive': [self.validation_end_exclusive],
                'validation_suspicious_transaction_count': [suspicious_transactions.height],
                'suspicious_component_count': [component_summary.height],
                'eligible_ring_count': [rings.height],
                'excluded_component_count': [
                    component_summary
                    .filter(
                        ~pl.col('is_eligible_ring')
                    )
                    .height
                ],
                'seed_account_count': [ring_tables['seeds'].height],
                'target_account_count': [ring_tables['targets'].height],
                'minimum_ring_size_observed': [
                    rings
                    .get_column(
                        'ring_size'
                    )
                    .min()
                ],
                'median_ring_size': [
                    rings
                    .get_column(
                        'ring_size'
                    )
                    .median()
                ],
                'maximum_ring_size_observed': [
                    rings
                    .get_column(
                        'ring_size'
                    )
                    .max()
                ],
                'mixed_typology_ring_count': [
                    rings
                    .get_column(
                        'is_mixed_typology_ring'
                    )
                    .sum()
                ]
            }
        )

    def build_reachability_summary(
            self, 
            use_persisted_graph: bool = True
    ) -> pl.DataFrame:
        """
        Measure whether withheld targets are graph-reachable from pooled seeds

        The diagnostic uses the label-free observation graph. Forward and reverse reachability indicate which PageRank
        direction can propagate seed mass. Weak reachability is the basic feasibility check
        """
        # validate
        if not isinstance(use_persisted_graph, bool):
            raise TypeError(
                'use_persisted_graph must be a boolean'
            )

        paths = self.get_experiment_paths()

        # graph tables
        if (
            use_persisted_graph
            & paths.nodes.is_file()
            & paths.edges.is_file()
        ):
            graph_tables = {
                'nodes': pl.scan_parquet(
                    source = paths.nodes
                ),
                'edges': pl.scan_parquet(
                    source = paths.edges
                )
            }
        else:
            graph_tables = self.build_observation_graph_tables()

        # ring tables
        ring_tables = self.build_ring_tables(
            force_rebuild = False
        )

        # graph counts
        graph_counts = (
            pl.collect_all(
                [
                    # nodes
                    graph_tables['nodes']
                    .select(
                        [
                            # node count
                            pl.len()
                            .alias(
                                'node_count'
                            ),
                            # maximum node id
                            pl.col('node_id')
                            .max()
                            .alias(
                                'maximum_node_id'
                            )
                        ]
                    ),
                    # edges
                    graph_tables['edges']
                    .select(
                        # edge count
                        pl.len()
                        .alias(
                            'edge_count'
                        )
                    )
                ],
                engine = 'streaming'
            )
        )

        # node count
        node_count = int(
            graph_counts[0]
            .get_column(
                'node_count'
            )
            .item()
        )

        # maximum node id
        maximum_node_id = int(
            graph_counts[0]
            .get_column(
                'maximum_node_id'
            )
            .item()
        )

        # edge pairs
        edge_pairs = (
            graph_tables['edges']
            .select(
                [
                    'source_node_id',
                    'target_node_id'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

        # igraph
        graph = ig.Graph(
            n = node_count,
            edges = [
                (int(source_node_id), int(target_node_id))
                for source_node_id, target_node_id in edge_pairs.iter_rows()
            ],
            directed = True
        )

        # seed node ids
        seed_node_ids = [
            int(node_id)
            for node_id in (
                ring_tables['seeds']
                .get_column(
                    'node_id'
                )
                .to_list()
            )
        ]

        # target node ids
        target_node_ids = [
            int(node_id)
            for node_id in (
                ring_tables['targets']
                .get_column(
                    'node_id'
                )
                .to_list()
            )
        ]

        # forward reachable nodes
        forward_reachable = self._multi_source_reachable_nodes(
            graph = graph,
            seed_node_ids = seed_node_ids,
            direction = 'forward'
        )

        # reverse reachable nodes
        reverse_reachable = self._multi_source_reachable_nodes(
            graph = graph,
            seed_node_ids = seed_node_ids,
            direction = 'reverse'
        )

        # weakly reachable nodes
        weakly_reachable = self._multi_source_reachable_nodes(
            graph = graph,
            seed_node_ids = seed_node_ids,
            direction = 'weak'
        )

        # forward targets
        forward_targets = set(target_node_ids) & forward_reachable

        # reverse targets
        reverse_targets = set(target_node_ids) & reverse_reachable

        # weak targets
        weak_targets = set(target_node_ids) & weakly_reachable

        # undirected targets
        either_direction_targets = (
            set(target_node_ids) & (forward_reachable | reverse_reachable)
        )

        # target count
        target_count = len(target_node_ids)

        return pl.DataFrame(
            {
                'observation_node_count': [node_count],
                'observation_edge_count': [
                    int(
                        graph_counts[1]
                        .get_column(
                            'edge_count'
                        )
                        .item()
                    )
                ],
                'weakly_reachable_target_count': [len(weak_targets)],
                'weakly_reachable_target_share': [
                    len(weak_targets) / target_count
                ],
                'forward_reachable_target_count': [len(forward_targets)],
                'forward_reachable_target_share': [
                    len(forward_targets) / target_count
                ],
                'reverse_reachable_target_count': [len(reverse_targets)],
                'reverse_reachable_target_share': [
                    len(reverse_targets) / target_count
                ],
                'either_direction_reachable_target_count': [len(either_direction_targets)],
                'either_direction_reachable_target_share': [
                    len(either_direction_targets) / target_count
                ]
            }
        )


    ### public persistence methods
    def get_experiment_paths(self) -> RingExperimentPaths:
        """ Return the standard artifact patahs for this experiment """
        # validate output_dir is a Path instance
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        return RingExperimentPaths(
            directory = self.output_dir,
            nodes = self.output_dir / 'nodes.parquet',
            edges = self.output_dir / 'edges.parquet',
            rings = self.output_dir / 'rings.parquet',
            ring_accounts = self.output_dir / 'ring_accounts.parquet',
            seeds = self.output_dir / 'seeds.parquet',
            targets = self.output_dir / 'targets.parquet',
            manifest = self.output_dir / 'manifest.json'
        )

    def validate_experiment(self) -> None:
        """ Validate the files of an existing ring experiment """
        # experiment paths
        paths = self.get_experiment_paths()

        # validate parquet files
        for table_name, path in paths.parquet_paths().items():
            self._validate_parquet_file(
                table_name = table_name,
                path = path
            )

        # validate if ring-experiment manifest exists
        if not paths.manifest.is_file():
            raise FileNotFoundError(
                f'Ring-experiment manifest does not exist: {paths.manifest}'
            )

        # validate if ring-experiment manifest is empty on disk
        if paths.manifest.stat().st_size == 0:
            raise ValueError(
                f'Ring-experiment manifest is empty: {paths.manifest}'
            )

    def scan_experiment(self) -> dict[str, pl.LazyFrame]:
        """ Lazily scan the persisted Parquet experiment tables """
        # validate files of an existing ring experiment
        self.validate_experiment()

        # experiment paths
        paths = self.get_experiment_paths()

        return {
            table_name: pl.scan_parquet(source = path)
            for table_name, path in paths.parquet_paths().items()
        }

    def write_experiment(
            self,
            overwrite: bool = False
    ) -> RingExperimentPaths:
        """ Build and persist the validation ring-expansion experiment """
        # validate if overwrite is a boolean instance
        if not isinstance(overwrite, bool):
            raise TypeError(
                'overwrite must be a boolean'
            )

        # experiment paths
        paths = self.get_experiment_paths()

        # experiment paths as dict
        final_paths = paths.as_dict()

        # artifacts
        existing_artifacts = [
            artifact_name
            for artifact_name, path in final_paths.items()
            if path.exists()
        ]

        if existing_artifacts and not overwrite:
            if len(existing_artifacts) == len(final_paths):
                # validate ring experiment
                self.validate_experiment()

                return paths

            # sorted existing artifacts
            sorted_existing_artifacts = ', '.join(
                sorted(existing_artifacts)
            )

            raise FileExistsError(
                'The ring experiment is incomplete. Existing artifacts: '
                f'{sorted_existing_artifacts}. Re-run with overwrite=True to rebuild it'
            )

        # create paths directory
        paths.directory.mkdir(
            parents = True,
            exist_ok = True
        )

        temporary_paths = {
            artifact_name: self._get_temporary_path(path = path)
            for artifact_name, path in final_paths.items()
        }

        for temporary_path in temporary_paths.values():
            # unlink temporary path
            temporary_path.unlink(
                missing_ok = True
            )

        try:
            graph_tables = self.build_observation_graph_tables()
            ring_tables = self.build_ring_tables(
                force_rebuild = False
            )

            # experiment tables
            experiment_tables: dict[str, pl.LazyFrame] = {
                **graph_tables,
                **{
                    table_name: table.lazy()
                    for table_name, table in ring_tables.items()
                }
            }

            # sink and validate parquet files in experiment tables
            for table_name, lazy_frame in experiment_tables.items():
                self._sink_parquet(
                    lazy_frame = lazy_frame,
                    path = temporary_paths[table_name]
                )

                self._validate_parquet_file(
                    table_name = table_name,
                    path = temporary_paths[table_name]
                )

            # experiment summary
            summary = (
                self.build_experiment_summary()
                .to_dicts()[0]
            )

            # define manifest
            manifest = {
                'experiment': 'validation_ring_expansion',
                'ring_definition': (
                    'weak component of validation-window suspicious transaction graph'
                ),
                'seed_selection': (
                    'earliest suspicious account appearance, then account identifier'
                ),
                'validation_start': self.validation_start.isoformat(),
                'validation_end_exclusive': self.validation_end_exclusive.isoformat(),
                'observation_graph_cutoff_exclusive': self.validation_end_exclusive.isoformat(),
                'minimum_ring_size': self.minimum_ring_size,
                'seed_fraction': self.seed_fraction,
                'minimum_seed_count': self.minimum_seed_count,
                'summary': {
                    key: value
                    for key, value in summary.items()
                    if key not in {
                        'validation_start',
                        'validation_end_exclusive'
                    }
                },
                'label_separation': (
                    'is_laundering and laundering_type are absent from nodes.parquet and edges.parquet'
                )
            }

            # dump manifest
            temporary_paths['manifest'].write_text(
                json.dumps(
                    manifest,
                    indent = 2
                ),
                encoding = 'utf-8'
            )

            # replace temporary file paths with final paths
            for artifact_name, final_path in final_paths.items():
                temporary_paths[artifact_name].replace(final_path)

            # validate experiment
            self.validate_experiment()

        except Exception:
            for temporary_path in temporary_paths.values():
                # unlink temporary path
                temporary_path.unlink(
                    missing_ok = True
                )

            raise

        return paths