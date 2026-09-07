from pathlib import Path

import polars as pl
import igraph as ig

from igraph.clustering import VertexClustering

class SAMLDGraphEDA:
    """
    Analyze one persisted SAML-D graph snapshot
    
    Polars is used as table-level summaries, while igraph is used for topology that requires an in-memory graph,
    such as connected components and reciprocity
    
    Future targets are used only for validation diagnostics, never as graph inputs
    """

    # define class-level snapshot file names
    SNAPSHOT_FILE_NAMES: dict[str, str] = {
          'nodes': 'nodes.parquet',
          'edges': 'edges.parquet',
          'seeds': 'seeds.parquet',
          'targets': 'targets.parquet'
    }

    # specifying minimum columns each persisted table must contain
    REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
        'nodes': (
            'node_id',
            'account',
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
        ),
        'edges': (
            'source_node_id',
            'target_node_id',
            'transaction_count',
            'count_weight',
            'log_count_weight',
            'active_day_count',
            'cross_currency_share'
        ),
        'seeds': (
            'node_id',
            'account'
        ),
        'targets': (
            'node_id',
            'account',
            'known_before_window',
            'seen_before_window',
            'is_new_suspicious_account',
            'is_rankable_new_suspicious_account'
        )
    }

    # node related graph evaluation metrics
    NODE_METRICS: tuple[str, ...] = (
        'outgoing_transaction_count',
        'incoming_transaction_count',
        'account_transaction_event_count',
        'out_degree',
        'in_degree',
        'total_directed_degree',
        'bank_location_count'
    )

    # edge related graph evaluation metrics
    EDGE_METRICS: tuple[str, ...] = (
        'transaction_count',
        'log_count_weight',
        'active_day_count',
        'cross_currency_share'
    )

    # top node related graph evaluation metrics
    TOP_NODE_METRICS: tuple[str, ...] = (
        'outgoing_transaction_count',
        'incoming_transaction_count',
        'account_transaction_event_count',
        'out_degree',
        'in_degree',
        'total_directed_degree'
    )

    def __init__(
            self,
            snapshot_dir: Path | str
    ) -> None:
        # snapshot directory
        self.snapshot_dir = Path(snapshot_dir)

        # validate and lazily scan persisted tables
        self.snapshot = self._scan_snapshot()
        self.nodes = self.snapshot['nodes']
        self.edges = self.snapshot['edges']
        self.seeds = self.snapshot['seeds']
        self.targets = self.snapshot['targets']

        # in-memory graph and component caches
        self._graph: ig.Graph | None = None
        self._component_cache: dict[str, VertexClustering] = {}
    
    ### private helper methods
    def _get_snapshot_paths(self) -> dict[str, Path]:
        """ Return the four expected Parquet paths """
        return {
            table_name: self.snapshot_dir / file_name
            for table_name, file_name in self.SNAPSHOT_FILE_NAMES.items()
        }

    @classmethod
    def _validate_parquet_schema(
        cls,
        table_name: str,
        path: Path
    ) -> None:
        """ Validate the minimum schema required for graph EDA """
        # check if snapshot table exists
        if not path.is_file():
            raise FileNotFoundError(
                F'Persisted snapshot table does not exist: {path}'
            )

        # check the file size of path
        if path.stat().st_size == 0:
            raise ValueError(
                f'Persisted snapshot table is empty: {path}'
            )

        # get actual columns from the parquet file schema
        actual_columns = set(
            pl.scan_parquet(source = path)
            .collect_schema()
            .names()
        )

        # expected columns
        required_columns = set(
            cls.REQUIRED_COLUMNS[table_name]
        )

        # missing columns that are not present in the actual columns
        missing_columns = required_columns - actual_columns

        # check the missing columns, and return sorted missing columns if present
        if missing_columns:
            sorted_missing_columns = ', '.join(
                sorted(missing_columns)
            )

            raise ValueError(
                f'{table_name!r} is missing graph-EDA columns: '
                f'{sorted_missing_columns}'
            )

    def _scan_snapshot(self) -> dict[str, pl.LazyFrame]:
        """ Validate and lazily scan the persisted snapshot """
        # get parquet snapshot paths
        paths = self._get_snapshot_paths()

        # validate snapshot paths
        for table_name, path in paths.items():
            # validate parquet schema
            self._validate_parquet_schema(
                table_name = table_name,
                path = path
            )

        return {
            table_name: pl.scan_parquet(source = path)
            for table_name, path in paths.items()
        }

    @staticmethod
    def _validate_top_n(
        top_n: int
    ) -> None:
        """ Validate a requested result-table size """
        # validate top_n instance type
        if not isinstance(top_n, int):
            raise TypeError(
                'top_n must be an integer.'
            )

        # logical expression on top_n
        if top_n <= 0:
            raise ValueError(
                'top_n must be a non-negative integer.'
            )

    @staticmethod
    def _validate_component_mode(
        mode: str
    ) -> None:
        """ Validate an igraph connected-component mode """
        # check if mode is either weak or strong
        if mode not in {'weak', 'strong'}:
            raise ValueError(
                "mode must be either 'weak' or 'strong'"
            )

    @staticmethod
    def _build_numeric_summary(
        lazy_frame: pl.LazyFrame,
        metrics: tuple[str, ...]
    ) -> pl.DataFrame:
        """ Build a long-form distribution summary with one source scan """
        # define numeric summary
        summary_expressions: list[pl.Expr] = []

        for metric in metrics:
            # loop over each metric
            metric_expression = (
                pl.col(metric)
                .cast(pl.Float64)
            )

            # append to summary expressions
            summary_expressions.extend(
                [
                    # minimum
                    metric_expression
                    .min()
                    .alias(
                        f'{metric}__minimum'
                    ),
                    # mean
                    metric_expression
                    .mean()
                    .alias(
                        f'{metric}__mean'
                    ),
                    # median
                    metric_expression
                    .median()
                    .alias(
                        f'{metric}__median'
                    ),
                    # q_90
                    metric_expression
                    .quantile(0.90)
                    .alias(
                        f'{metric}__p90'
                    ),
                    # q_95
                    metric_expression
                    .quantile(0.95)
                    .alias(
                        f'{metric}__p95'
                    ),
                    # q_99
                    metric_expression
                    .quantile(0.99)
                    .alias(
                        f'{metric}__p99'
                    ),
                    # maximum
                    metric_expression
                    .max()
                    .alias(
                        f'{metric}__maximum'
                    ),
                    # zero share
                    (
                        metric_expression == 0
                    )
                    .cast(pl.Float64)
                    .mean()
                    .alias(
                        f'{metric}__zero_share'
                    )
                ]
            )

        # long summary
        long_summary = (
            lazy_frame
            .select(
                summary_expressions
            )
            .collect(
                engine = 'streaming'
            )
            .row(
                index = 0,
                named = True
            )
        )

        return pl.DataFrame(
            [
                {
                    'metric': metric,
                    'minimum': long_summary[f'{metric}__minimum'],
                    'mean': long_summary[f'{metric}__mean'],
                    'median': long_summary[f'{metric}__median'],
                    'p90': long_summary[f'{metric}__p90'],
                    'p95': long_summary[f'{metric}__p95'],
                    'p99': long_summary[f'{metric}__p99'],
                    'maximum': long_summary[f'{metric}__maximum'],
                    'zero_share': long_summary[f'{metric}__zero_share']
                }
                for metric in metrics
            ]
        )

    def _get_components(
            self,
            mode: str
    ) -> VertexClustering:
        """ Return cached weakk or strong connected components """
        # validate component mode
        self._validate_component_mode(
            mode = mode
        )

        # build connected component for a given mode if not cached before
        if mode not in self._component_cache:
            # build igraph
            graph = self.build_igraph()

            # dump connected components into component cache for a given mode
            self._component_cache[mode] = (
                graph.connected_components(
                    mode = mode
                )
            )

        return self._component_cache[mode]

    def _build_node_population_table(self) -> pl.LazyFrame:
        """ Attach validation-only seed and rankable-target populations """
        # seed nodes
        seed_nodes = (
            self.seeds
            .select(
                'node_id'
            )
            .with_columns(
                pl.lit(True)
                .alias(
                    'is_seed'
                )
            )
        )

        # rankable target nodes
        rankable_target_nodes = (
            self.targets
            .filter(
                pl.col('is_rankable_new_suspicious_account')
            )
            .select(
                'node_id'
            )
            .with_columns(
                pl.lit(True)
                .alias(
                    'is_rankable_target'
                )
            )
        )

        return (
            self.nodes
            .join(
                seed_nodes,
                on = 'node_id',
                how = 'left'
            )
            .join(
                rankable_target_nodes,
                on = 'node_id',
                how = 'left'
            )
            .with_columns(
                [
                    pl.col('is_seed')
                    .fill_null(False),
                    pl.col('is_rankable_target')
                    .fill_null(False)
                ]
            )
            .with_columns(
                # population
                pl.when(
                    pl.col('is_seed')
                )
                .then(
                    pl.lit('historical_seed')
                )
                .when(
                    pl.col('is_rankable_target')
                )
                .then(
                    pl.lit('rankable_validation_target')
                )
                .otherwise(
                    pl.lit('other_candidate')
                )
                .alias(
                    'population'
                )
            )
        )
        

    ### public table-level EDA methods
    def get_table_summary(self) -> pl.DataFrame:
        """ Return row counts for persisted validation tables """
        return (
            pl.concat(
                [
                    lazy_frame
                    .select(
                        [
                            # table name
                            pl.lit(table_name)
                            .alias(
                                'table'
                            ),
                            # row count
                            pl.len()
                            .cast(pl.UInt64)
                            .alias(
                                'row_count'
                            )
                        ]
                    )
                    for table_name, lazy_frame in self.snapshot.items()
                ],
                how = 'vertical'
            )
            .collect(
                engine = 'streaming'
            )
        )

    def get_supervision_summary(self) -> pl.DataFrame:
        """ Summarize PageRank seeds, candidates and future targets """
        # node counts
        node_counts = (
            self.nodes
            .select(
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'node_count'
                )
            )
        )

        # seed counts
        seed_counts = (
            self.seeds
            .select(
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'seed_count'
                )
            )
        )

        # target counts
        target_counts = (
            self.targets
            .select(
                [
                    # future suspicious account count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'future_suspicious_account_count'
                    ),
                    # previously known future account count
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
                    # rankable target count
                    pl.col('is_rankable_new_suspicious_account')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'rankable_target_count'
                    ),
                    # unrankable target count
                    pl.col('node_id')
                    .null_count()
                    .cast(pl.UInt64)
                    .alias(
                        'unrankable_target_count'
                    )
                ]
            )
        )

        return (
            node_counts
            .join(
                seed_counts,
                how = 'cross'
            )
            .join(
                target_counts,
                how = 'cross'
            )
            .with_columns(
                # candidate count
                (
                    pl.col('node_count') - pl.col('seed_count')
                )
                .alias(
                    'candidate_count'
                )
            )
            .with_columns(
                # seed share
                (
                    pl.col('seed_count') / pl.col('node_count')
                )
                .alias(
                    'seed_share'
                ),
                # rankable target coverage
                (
                    pl.col('rankable_target_count') / pl.col('new_future_suspicious_account_count')
                )
                .alias(
                    'rankable_target_coverage'
                ),
                # candidate positive rate
                (
                    pl.col('rankable_target_count') / pl.col('candidate_count')
                )
                .alias(
                    'candidate_positive_rate'
                )
            )
            .select(
                [
                    'node_count',
                    'seed_count',
                    'candidate_count',
                    'seed_share',
                    'future_suspicious_account_count',
                    'previously_known_future_account_count',
                    'new_future_suspicious_account_count',
                    'rankable_target_count',
                    'unrankable_target_count',
                    'rankable_target_coverage',
                    'candidate_positive_rate'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

    def get_node_role_summary(self) -> pl.DataFrame:
        """ Summarize sender-only, receiver-only and mixed-role accounts """
        return (
            self.nodes
            .with_columns(
                # node role
                pl.when(
                    pl.col('has_outgoing_activity')
                    & pl.col('has_incoming_activity')
                )
                .then(
                    pl.lit('sender_and_receiver')
                )
                .when(
                    pl.col('has_outgoing_activity')
                )
                .then(
                    pl.lit('sender_only')
                )
                .otherwise(
                    pl.lit('receiver_only')
                )
                .alias(
                    'node_role'
                )
            )
            .group_by(
                'node_role'
            )
            .agg(
                # node count per node role
                pl.len()
                .cast(pl.UInt64)
                .alias(
                    'node_count'
                )
            )
            .with_columns(
                # node share per node role
                (
                    pl.col('node_count') / pl.col('node_count').sum()
                )
                .alias(
                    'node_share'
                )
            )
            .with_columns(
                # role order
                pl.when(
                    pl.col('node_role') == 'sender_and_receiver'
                )
                .then(
                    0
                )
                .when(
                    pl.col('node_role') == 'sender_only'
                )
                .then(
                    1
                )
                .otherwise(
                    2
                )
                .alias(
                    '_role_order'
                )
            )
            .sort(
                '_role_order'
            )
            .drop(
                '_role_order'
            )
            .collect(
                engine = 'streaming'
            )
        )

    def get_node_metric_summary(self) -> pl.DataFrame:
        """ Return distribution statistics for node activity and degree """
        return self._build_numeric_summary(
            lazy_frame = self.nodes,
            metrics = self.NODE_METRICS
        )

    def get_edge_metric_summary(self) -> pl.DataFrame:
        """ Return distribution statistics for edge activity and weights """
        return self._build_numeric_summary(
            lazy_frame = self.edges,
            metrics = self.EDGE_METRICS
        )

    def get_population_structural_summary(self) -> pl.DataFrame:
        """ Compare seeds, rankable targets and remaining candidates """
        return (
            self._build_node_population_table()
            .group_by(
                'population'
            )
            .agg(
                [
                    # node count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'node_count'
                    ),
                    # mean transaction event count
                    pl.col('account_transaction_event_count')
                    .mean()
                    .alias(
                        'mean_transaction_event_count'
                    ),
                    # mean total directed degree
                    pl.col('total_directed_degree')
                    .mean()
                    .alias(
                        'mean_total_directed_degree'
                    ),
                    # median total directed degree
                    pl.col('total_directed_degree')
                    .median()
                    .alias(
                        'median_total_directed_degree'
                    ),
                    # q95 total directed degree
                    pl.col('total_directed_degree')
                    .quantile(0.95)
                    .alias(
                        'p95_total_directed_degree'
                    ),
                    # sender and receiver share
                    (
                        pl.col('has_outgoing_activity')
                        & pl.col('has_incoming_activity')
                    )
                    .cast(pl.Float64)
                    .mean()
                    .alias(
                        'sender_and_receiver_share'
                    ),
                    # receiver-only share
                    (
                        ~pl.col('has_outgoing_activity')
                        & pl.col('has_incoming_activity')
                    )
                    .cast(pl.Float64)
                    .mean()
                    .alias(
                        'receiver_only_share'
                    ),
                    # sender-only share
                    (
                        pl.col('has_outgoing_activity')
                        & ~pl.col('has_incoming_activity')
                    )
                    .cast(pl.Float64)
                    .mean()
                    .alias(
                        'sender_only_share'
                    ),
                    # multi location share
                    pl.col('is_multi_location')
                    .cast(pl.Float64)
                    .mean()
                    .alias(
                        'multi_location_share'
                    )
                ]
            )
            .with_columns(
                pl.when(
                    pl.col('population') == 'historical_seed'
                )
                .then(
                    0
                )
                .when(
                    pl.col('population') == 'rankable_validation_target'
                )
                .then(
                    1
                )
                .otherwise(
                    2
                )
                .alias(
                    '_population_order'
                )
            )
            .sort(
                '_population_order'
            )
            .drop(
                '_population_order'
            )
            .collect(
                engine = 'streaming'
            )
        )

    def get_top_nodes(
            self,
            metric: str,
            top_n: int = 20
    ) -> pl.DataFrame:
        """ Return the highest-ranked historical nodes for one metric """
        # validate if metric is in self.TOP_NODE_METRICS
        if metric not in self.TOP_NODE_METRICS:
            # valid metrics within self.TOP_NODE_METRICS
            valid_metrics = ', '.join(
                self.TOP_NODE_METRICS
            )

            raise ValueError(
                f'Unsupported node metric: {metric}. '
                f'Choose from: {valid_metrics}'
            )

        # validate top_n instance type and logical expression
        self._validate_top_n(
            top_n = top_n
        )

        # columns to be selected in the node-population table
        selected_columns = list(
            dict.fromkeys(
                [
                    'node_id',
                    'account',
                    metric,
                    'out_degree',
                    'in_degree',
                    'total_directed_degree',
                    'outgoing_transaction_count',
                    'incoming_transaction_count',
                    'population'
                ]
            )
        )

        return (
            self._build_node_population_table()
            .sort(
                [
                    metric,
                    'node_id'
                ],
                descending = [
                    True,
                    False
                ]
            )
            .select(
                selected_columns
            )
            .head(
                top_n
            )
            .collect(
                engine = 'streaming'
            )
        )


    ### public igraph-level EDA methods
    def build_igraph(
            self,
            force_rebuild: bool = False
    ) -> ig.Graph:
        """  
        Materialize the directed validation graph in igraph

        igraph vertex indices equal snapshot node IDs because the persisted node IDs are 
        contiguous and zero-based
        """
        # if graph is already constructed, return cached graph
        if self._graph is not None and not force_rebuild:
            return self._graph

        # node count
        node_count = (
            self.nodes
            .select(
                pl.len()
            )
            .collect(
                engine = 'streaming'
            )
            .item()
        )

        # edge frame
        edge_frame = (
            self.edges
            .select(
                [
                    'source_node_id',
                    'target_node_id',
                    'count_weight',
                    'log_count_weight'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

        # edge pairs
        edge_pairs = (
            edge_frame
            .select(
                [
                    'source_node_id',
                    'target_node_id'
                ]
            )
            .rows()
        )

        # igraph object
        graph = ig.Graph(
            n = node_count,
            edges = edge_pairs,
            directed = True
        )

        # igraph count weights
        graph.es['count_weight'] = (
            edge_frame
            .get_column(
                'count_weight'
            )
            .to_list()
        )

        # igraph log count weights
        graph.es['log_count_weight'] = (
            edge_frame
            .get_column(
                'log_count_weight'
            )
            .to_list()
        )

        # dump igraph into cache, and clear component cache
        self._graph = graph
        self._component_cache.clear()

        return graph

    def get_connectivity_summary(self) -> pl.DataFrame:
        """ Summarize full directed-graph connectivity """
        # materialize the directed validation graph in igraph
        graph = self.build_igraph(
            force_rebuild = False
        )

        # get cached components
        weak_components = self._get_components(
            mode = 'weak'
        )
        strong_components = self._get_components(
            mode = 'strong'
        )

        # component sizes
        weak_sizes = weak_components.sizes()
        strong_sizes = strong_components.sizes()

        # in- and out-degrees
        in_degrees = graph.indegree(
            loops = False
        )
        out_degrees = graph.outdegree(
            loops = False
        )

        # node count
        node_count = graph.vcount()

        # edge count
        edge_count = graph.ecount()

        # isolated node count
        isolated_node_count = sum(
            in_degree == 0 and out_degree == 0
            for in_degree, out_degree in zip(
                in_degrees,
                out_degrees,
                strict = True
            )
        )

        # receiver-only node count
        receiver_only_node_count = sum(
            in_degree > 0 and out_degree == 0
            for in_degree, out_degree in zip(
                in_degrees,
                out_degrees,
                strict = True
            )
        )

        # sender-only node count
        sender_only_node_count = sum(
            in_degree == 0 and out_degree > 0
            for in_degree, out_degree in zip(
                in_degrees,
                out_degrees,
                strict = True
            )
        )

        return pl.DataFrame(
            {
                'node_count': [node_count],
                'edge_count': [edge_count],
                'density': [
                    graph.density(
                        loops = False
                    )
                ],
                'reciprocity': [
                    graph.reciprocity(
                        ignore_loops = True
                    )
                ],
                'mean_in_degree': [
                    edge_count / node_count
                ],
                'mean_out_degree': [
                    edge_count / node_count
                ],
                'isolated_node_count': [isolated_node_count],
                'receiver_only_node_count': [receiver_only_node_count],
                'sender_only_node_count': [sender_only_node_count],
                'weak_component_count': [len(weak_sizes)],
                'largest_weak_component_size': [max(weak_sizes)],
                'largest_weak_component_share': [
                    max(weak_sizes) / node_count
                ],
                'strong_component_count': [len(strong_sizes)],
                'largest_strong_component_size': [max(strong_sizes)],
                'largest_strong_component_share': [
                    max(strong_sizes) / node_count
                ]
            }
        )

    def get_component_size_summary(
            self,
            mode: str,
            top_n: int = 20
    ) -> pl.DataFrame:
        """ Return the largest weak or strong component sizes of a selected mode """
        # validate an igraph component mode
        self._validate_component_mode(
            mode = mode
        )

        # validate top_n instance type and logical expression
        self._validate_top_n(
            top_n = top_n
        )

        # materialize the directed validation graph in igraph
        graph = self.build_igraph(
            force_rebuild = False
        )

        # get cached components
        components = self._get_components(
            mode = mode
        )

        # component sizes of top_n 
        component_sizes = sorted(
            components.sizes(),
            reverse = True
        )[:top_n]

        # component summary
        cumulative_size = 0
        summary_rows: list[dict[str, int | float | str]] = []

        for component_rank, component_size in enumerate(component_sizes, start = 1):

            cumulative_size += component_size
            summary_rows.append(
                {
                    'component_mode': mode,
                    'component_rank': component_rank,
                    'component_size': component_size,
                    'node_share': component_size / graph.vcount(),
                    'cumulative_node_share': cumulative_size / graph.vcount()
                }
            )

        return pl.DataFrame(summary_rows)