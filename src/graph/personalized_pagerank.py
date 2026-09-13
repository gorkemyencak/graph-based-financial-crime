import numpy as np
import polars as pl
import igraph as ig

from pathlib import Path
from collections.abc import Sequence

class SAMLDRingPersonalizedPageRank:
    """
    Score ad evaluate the persisted SAML-D ring-expansion experiment

    The class keeps scoring and evaluation deliberately separate:
        - nodes, edges, and known seeds are used to compute scores
        - held-out targets are attached only after every score has been computed
        - the persisted score table never contains target labels or ring IDs

    Personalized PageRank uses all known ring seeds as one pooled restart population. The default uniform_seed scheme assigns
    equal restart mass to every seed. ring_balanced assigns equal total restart mass to every approximate ring and divides 
    that mass equally among its seeds. Matched forwad and reverse vectors are also averaged into bidirectional scores so that 
    either transaction-flow orientation can contribute to a ranking    
    """

    EXPERIMENT_FILE_NAMES: dict[str, str] = {
        'nodes': 'nodes.parquet',
        'edges': 'edges.parquet',
        'rings': 'rings.parquet',
        'ring_accounts': 'ring_accounts.parquet',
        'seeds': 'seeds.parquet',
        'targets': 'targets.parquet'
    }

    REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
        'nodes': (
            'node_id',
            'account',
            'out_degree',
            'in_degree',
            'total_directed_degree',
            'account_transaction_event_count'
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
            'account'
        ),
        'targets': (
            'ring_id',
            'node_id',
            'account'
        )
    }

    REFERENCE_SCORE_COLUMNS: tuple[str, ...] = (
        'out_degree',
        'in_degree',
        'total_directed_degree',
        'account_transaction_event_count'
    )

    ORDINARY_PAGERANK_CONFIGURATIONS: dict[str, tuple[str, str | None]] = {
        'pagerank_unweighted': (
            'forward',
            None
        ),
        'pagerank_count_weighted': (
            'forward',
            'count_weight'
        ),
        'pagerank_log_count_weighted': (
            'forward',
            'log_count_weighted'
        ),
        'reverse_pagerank_unweighted': (
            'reverse',
            None
        ),
        'reverse_pagerank_count_weighted': (
            'reverse',
            'count_weight'
        ),
        'reverse_pagerank_log_count_weighted': (
            'reverse',
            'log_count_weighted'
        )
    }

    BIDIRECTIONAL_ORDINARY_PAGERANK_COLUMNS: dict[str, tuple[str, str]] = {
        'bidirectional_pagerank_unweighted': (
            'pagerank_unweighted',
            'reverse_pagerank_unweighted'
        ),
        'bidirectional_pagerank_count_weighted': (
            'pagerank_count_weighted',
            'reverse_pagerank_count_weighted'
        ),
        'bidirectional_pagerank_log_count_weighted': (
            'pagerank_log_count_weighted',
            'reverse_pagerank_log_count_weighted'
        )
    }

    PERSONALIZED_PAGERANK_CONFIGURATIONS: dict[str, tuple[str, str | None]] = {
        'personalized_pagerank_unweighted': (
            'forward',
            None
        ),
        'personalized_pagerank_count_weighted': (
            'forward',
            'count_weight'
        ),
        'personalized_pagerank_log_count_weighted': (
            'forward',
            'log_count_weight'
        ),
        'reverse_personalized_pagerank_unweighted': (
            'reverse',
            None
        ),
        'reverse_personalized_pagerank_count_weighted': (
            'reverse',
            'count_weight'
        ),
        'reverse_personalized_pagerank_log_count_weighted': (
            'reverse',
            'log_count_weight'
        )
    }

    BIDIRECTIONAL_PERSONALIZED_PAGERANK_COLUMNS: dict[str, tuple[str, str]] = {
        'bidirectional_personalized_pagerank_unweighted': (
            'personalized_pagerank_unweighted',
            'reverse_personalized_pagerank_unweighted'
        ),
        'bidirectional_personalized_pagerank_count_weighted': (
            'personalized_pagerank_count_weighted',
            'reverse_personalized_pagerank_count_weighted'
        ),
        'bidirectional_personalized_pagerank_log_count_weighted': (
            'personalized_pagerank_log_count_weighted',
            'reverse_personalized_pagerank_log_count_weighted'
        )
    }

    PERSONALIZATION_SCHEMES: tuple[str, ...] = (
        'uniform_seed',
        'ring_balanced'
    )

    def __init__(
            self,
            experiment_dir: Path | str,
            damping: float | int = 0.85,
            personalization_scheme: str = 'uniform_seed'
    ) -> None:
        # validate damping factor
        self._validate_damping(
            damping = damping
        )

        # validate personalization scheme
        self._validate_personalization_scheme(
            personalization_scheme = personalization_scheme
        )

        # attributes
        self.experiment_dir = Path(experiment_dir)
        self.damping = float(damping)
        self.personalization_scheme = personalization_scheme

        # experiment attributes
        self.experiment = self._scan_experiment()
        self.nodes = self.experiment['nodes']
        self.edges = self.experiment['edges']
        self.rings = self.experiment['rings']
        self.ring_accounts = self.experiment['ring_accounts']
        self.seeds = self.experiment['seeds']
        self.targets = self.experiment['targets']

        # graph attributes
        self._graph: ig.Graph | None = None
        self._reverse_graph: ig.Graph | None = None
        self._base_node_table: pl.DataFrame | None = None
        self._seed_table: pl.DataFrame | None = None
        self._target_table: pl.DataFrame | None = None
        self._score_table: pl.DataFrame | None = None
        self._candidate_score_table: pl.DataFrame | None = None
        self._seed_node_ids: np.ndarray | None = None
        self._target_node_ids: np.ndarray | None = None
        self._reset_vector: np.ndarray | None = None
        self._reachability_cache: dict[str, np.ndarray] = {}
    
    ### private validation and loading methods
    @staticmethod
    def _validate_damping(
        damping: float | int
    ) -> None:
        """ Validate PageRank damping factor """
        # validate if damping is either int or float instance
        if not isinstance(damping, int | float):
            raise TypeError(
                'damping must be either int or float instance'
            )

        # validate the range of damping factor
        if not 0.0 < float(damping) < 1.0:
            raise ValueError(
                'damping must be strictly between 0.0 and 1.0'
            )

    @classmethod
    def _validate_personalization_scheme(
        cls,
        personalization_scheme: str
    ) -> None:
        """ Validate the pooled restart-mass scheme """
        # validate whether personalization_scheme is a string instance
        if not isinstance(personalization_scheme, str):
            raise TypeError(
                'personalization_scheme must be a string'
            )

        # validate if personalization_scheme is in PERSONALIZATION_SCHEMES
        if personalization_scheme not in cls.PERSONALIZATION_SCHEMES:

            sorted_valid_schemes = ', '.join(
                sorted(cls.PERSONALIZATION_SCHEMES)
            )

            raise ValueError(
                f'Unsupported personalizaton scheme. Choose from: {sorted_valid_schemes}'
            )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        parameter_name: str
    ) -> None:
        """ Validate a strictly positive integer parameter """
        # validate whether value is an int instance
        if not isinstance(value, int):
            raise TypeError(
                f'{parameter_name} must be an integer'
            )

        # validate if value is non-negative
        if value <= 0:
            raise ValueError(
                f'{parameter_name} must be greater than 0'
            )

    @classmethod
    def _validate_parquet_schema(
        cls,
        table_name: str,
        path: Path
    ) -> None:
        """ Validate one persisted ring-experiment table """
        # validate if path exists
        if not path.is_file():
            raise FileNotFoundError(
                f'Ring-experiment table does not exist: {path}'
            )

        # validate if path is empty on disk
        if path.stat().st_size == 0:
            raise ValueError(
                f'Ring-experiment table is empty: {path}'
            )

        # extract actual columns from schema
        actual_columns = set(
            pl.scan_parquet(source = path)
            .collect_schema()
            .names()
        )

        # expected columns
        expected_columns = set(
            cls.REQUIRED_COLUMNS[table_name]
        )

        # missing columns
        missing_columns = expected_columns - actual_columns

        # validate missing columns, return sorted if missing columns present
        if missing_columns:
            sorted_missing_columns = ', '.join(
                sorted(missing_columns)
            )

            raise ValueError(
                f'{table_name!r} is missing Personalized PageRank columns: {sorted_missing_columns}'
            )

    def _get_experiment_paths(self) -> dict[str, Path]:
        """ Return the expected ring-experiment Parquet paths """
        return {
            table_name: self.experiment_dir / file_name
            for table_name, file_name in self.EXPERIMENT_FILE_NAMES.items()
        }

    def _scan_experiment(self) -> dict[str, pl.LazyFrame]:
        """ Validate and lazily scan the persisted ring experiment """
        # extract experiment paths
        paths = self._get_experiment_paths()

        for table_name, path in paths.items():
            # validate Parquet schema
            self._validate_parquet_schema(
                table_name = table_name,
                path = path
            )

        return {
            table_name: pl.scan_parquet(source = path)
            for table_name, path in paths.items()
        }

    @classmethod
    def _validate_score_columns(
        cls,
        score_columns: str | Sequence[str]
    ): #-> tuple[str, ...]:
        """ Validate and deduplicate requested ranking columns """
        # validate if score columns is a string instance
        if isinstance(score_columns, str):
            # if score is a single metric, convert to tuple
            requested_columns = (score_columns, )
        else:
            # if provided as sequence, then convert to tuple
            requested_columns = tuple(score_columns)

        # validate if requested columns is present
        if not requested_columns:
            raise ValueError(
                'At least one score column must be requested'
            )

        # valid score columns
        valid_columns = set(
            cls.get_score_columns()
        )

        # invalid score columns
        invalid_columns = (
            set(requested_columns) - valid_columns
        )

        # validate whether invalid score columns present, return sorted if exist
        if invalid_columns:
            sorted_invalid_columns = ', '.join(
                sorted(invalid_columns)
            )

            sorted_valid_columns = ', '.join(
                sorted(valid_columns)
            )

            raise ValueError(
                f'Unsupported score columns: {sorted_invalid_columns}. '
                f'Choose from: {sorted_valid_columns}'
            )

        # return requested columns, while removing duplicates and preserving order
        return tuple(
            dict.fromkeys(requested_columns)
        )

    @classmethod
    def _validate_k_values(
            cls,
            k_values: Sequence[int]
    ) -> tuple[int, ...]:
        """ Validate, deduplicate, and sort ranking cutoffs """
        # deduplicate and sort ranking cutoffs
        selected_k_values = tuple(
            sorted(
                set(k_values)
            )
        )

        # validate if k_values present
        if not selected_k_values:
            raise ValueError(
                'At least one k value must be requested'
            )

        # validate non-negativity of selected k values
        for k in selected_k_values:
            cls._validate_positive_integer(
                value = k,
                parameter_name = 'k_value'
            )

        return selected_k_values


    ### private graph and population methods
    def _collect_seed_table(self) -> pl.DataFrame:
        """ Collect unique seed accounts in deterministic order """
        if self._seed_table is None:
            # construct seed table
            seed_table = (
                self.seeds
                .select(
                    [
                        'ring_id',
                        'node_id',
                        'account'
                    ]
                )
                .unique(
                    subset = ['node_id'],
                    maintain_order = False
                )
                .sort(
                    [
                        'ring_id',
                        'account'
                    ]
                )
                .collect(
                    engine = 'streaming'
                )
            )

            # validate whether seed_table is empty
            if seed_table.is_empty():
                raise ValueError(
                    'Personalized PageRank requires at least one seed'
                )

            self._seed_table = seed_table

        return self._seed_table

    def _collect_target_table(self) -> pl.DataFrame:
        """ Collect unique held-out targets in deterministic order """
        if self._target_table is None:
            # construct target table
            target_table = (
                self.targets
                .select(
                    [
                        'ring_id',
                        'node_id',
                        'account',
                        'target_order'
                    ]
                )
                .unique(
                    subset = ['account'],
                    maintain_order = False
                )
                .sort(
                    [
                        'ring_id',
                        'target_order'
                    ]
                )
                .collect(
                    engine = 'streaming'
                )
            )

            # validate whether target_table is empty
            if target_table.is_empty():
                raise ValueError(
                    'Ring evaluation requires at least one target'
                )

            self._target_table = target_table

        return self._target_table

    def _get_seed_node_ids(self) -> np.ndarray:
        """ Return unique pooled seed vertex IDs """
        if self._seed_node_ids is None:
            # construct seed node ids
            seed_node_ids = (
                self._collect_seed_table()
                .get_column(
                    'node_id'
                )
                .to_numpy()
                .astype(
                    dtype = np.int64,
                    copy = False
                )
            )

            # validate seed node ids non-negativity
            if seed_node_ids.min() < 0:
                raise ValueError(
                    'Seed node IDs cannot be negative'
                )

            self._seed_node_ids = seed_node_ids

        return self._seed_node_ids

    def _get_target_node_ids(self) -> np.ndarray:
        """ Return unique held-out target vertex IDs """
        if self._target_node_ids is None:
            # construct target node ids
            target_node_ids = (
                self._collect_target_table()
                .get_column(
                    'node_id'
                )
                .to_numpy()
                .astype(
                    dtype = np.int64,
                    copy = False
                )
            )

            # validate target node ids must be non-negativity
            if target_node_ids.min() < 0:
                raise ValueError(
                    'Target node IDs cannot be negative'
                )

            self._target_node_ids = target_node_ids

        return self._target_node_ids

    def _build_base_node_table(self) -> pl.DataFrame:
        """ Collect node metrics in igraph vertex order and mark seeds """
        if self._base_node_table is None:
            # node table
            node_table = (
                self.nodes
                .select(
                    [
                        'node_id',
                        'account',
                        *self.REFERENCE_SCORE_COLUMNS
                    ]
                )
                .sort(
                    'node_id'
                )
                .collect(
                    engine = 'streaming'
                )
            )

            # node count
            node_count = node_table.height

            # node ids
            node_ids = (
                node_table
                .get_column(
                    'node_id'
                )
                .to_numpy()
                .astype(
                    dtype = np.int64,
                    copy = False
                )
            )

            # validate if node ids matching the node count
            if not np.array_equal(
                node_ids,
                np.arange(
                    node_count,
                    dtype = np.int64
                )
            ):
                raise ValueError(
                    'Observation node IDs must be contiguous, zero-based, '
                    'and equal to igraph vertex indices'
                )

            # seed node ids
            seed_node_ids = self._get_seed_node_ids()

            # target node ids
            target_node_ids = self._get_target_node_ids()

            # validate seed & target node ids falling in graph vertex range
            if seed_node_ids.max() >= node_count:
                raise ValueError(
                    'Seed node IDs fall outside the graph vertex range'
                )

            if target_node_ids.max() >= node_count:
                raise ValueError(
                    'Target node IDs fall outside the graph vertex range'
                )

            # validate disjoing property of seed & target node IDs
            if np.intersect1d(
                seed_node_ids,
                target_node_ids
            ).size > 0:
                raise ValueError(
                    'Seed and target node populations must be disjoint'
                )

            # mark seeds
            seed_mask = np.zeros(
                node_count,
                dtype = bool
            )

            seed_mask[seed_node_ids] = True

            self._base_node_table = (
                node_table
                .with_columns(
                    pl.Series(
                        name = 'is_seed',
                        values = seed_mask,
                        dtype = pl.Boolean
                    )
                )
            )            

        return self._base_node_table

    def build_igraph(
            self,
            force_rebuild: bool = False
    ) -> ig.Graph:
        """ Build and cache the weighted directed observation graph """
        # return igraph if it is already in the cache
        if self._graph is not None and not force_rebuild:
            return self._graph

        # node count
        node_count = (
            self._build_base_node_table()
            .height
        )

        # edge table
        edge_table = (
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

        # validate if edge table is empty
        if edge_table.is_empty():
            raise ValueError(
                'Personalized PageRank requires at least one graph edge'
            )

        # source node ids
        source_node_ids = (
            edge_table
            .get_column(
                'source_node_id'
            )
            .to_numpy()
            .astype(
                dtype = np.int64,
                copy = False
            )
        )

        # target node ids
        target_node_ids = (
            edge_table
            .get_column(
                'target_node_id'
            )
            .to_numpy()
            .astype(
                dtype = np.int64,
                copy = False
            )
        )

        # validate source & target nodes fall within graph vertex range
        if (
            source_node_ids.min() < 0
            or target_node_ids.min() < 0
            or source_node_ids.max() >= node_count
            or target_node_ids.max() >= node_count
        ):
            raise ValueError(
                'Observation edge endpoints fall outside the vertex range'
            )

        # construct directed igraph
        graph = ig.Graph(
            n = node_count,
            edges = list(
                zip(
                    source_node_ids.tolist(),
                    target_node_ids.tolist(),
                    strict = True
                )
            ),
            directed = True
        )

        # append count_weight and log_count_weight to igraph
        graph.es['count_weight'] = (
            edge_table
            .get_column(
                'count_weight'
            )
            .cast(pl.Float64)
            .to_list()
        )

        graph.es['log_count_weight'] = (
            edge_table
            .get_column(
                'log_count_weight'
            )
            .cast(pl.Float64)
            .to_list()
        )

        # validate count_weight and log_count_weight for finite and non-negative properties
        for weight_attribute in ('count_weight', 'log_count_weight'):

            weights = np.asarray(
                graph.es[weight_attribute],
                dtype = np.float64
            )

            if not np.isfinite(weights).all():
                raise ValueError(
                    f'{weight_attribute} contains infinite weights'
                )

            if np.any(weights <= 0):
                raise ValueError(
                    f'{weight_attribute} must contain strictly positive weights'
                )

        # assign graph attributes
        self._graph = graph
        self._reverse_graph = None
        self._reachability_cache.clear()

        return self._graph

    def _get_reverse_graph(self) -> ig.Graph:
        """ Build a cached graph with every transaction edge reversed """
        # build igraph if reverse graph is not in cache
        if self._reverse_graph is None:
            # construct igraph
            graph = self.build_igraph(
                force_rebuild = False
            )

            # reversed edges
            reversed_edges = [
                (target_node_id, source_node_id)
                for source_node_id, target_node_id in graph.get_edgelist()
            ]

            # construct reverse graph
            reverse_graph = ig.Graph(
                n = graph.vcount(),
                edges = reversed_edges,
                directed = True
            )

            # append count_weight and log_count_weight to reverse igraph
            reverse_graph.es['count_weight'] = list(
                graph.es['count_weight']
            )

            reverse_graph.es['log_count_weight'] = list(
                graph.es['log_count_weight']
            )

            # assing graph attribute
            self._reverse_graph = reverse_graph

        return self._reverse_graph

    def _build_reset_vector(self) -> np.ndarray:
        """ Build the configured pooled seed-personalization vector """
        # return reset vector if it is already constructed in the cache
        if self._reset_vector is not None:
            return self._reset_vector

        # node count
        node_count = (
            self.build_igraph()
            .vcount()
        )

        # seed table
        seed_table = self._collect_seed_table()

        # construct reset vector
        reset_vector = np.zeros(
            shape = node_count,
            dtype = np.float64
        )

        if self.personalization_scheme == 'uniform_seed':
            # seed node ids
            seed_node_ids = (
                seed_table
                .get_column(
                    'node_id'
                )
                .to_numpy()
                .astype(
                    dtype = np.int64,
                    copy = False
                )
            )

            reset_vector[seed_node_ids] = 1.0 / seed_node_ids.size

        else:
        # self.personalization_scheme == 'ring_balanced'
            # ring count
            ring_count = int(
                seed_table
                .get_column(
                    'ring_id'
                )
                .n_unique()
            )

            # validate if ring count is present
            if ring_count == 0:
                raise ValueError(
                    'Ring-balanced personalization requires at least one ring'
                )

            # seed weights
            seed_weights = (
                seed_table
                .with_columns(
                    # ring seed count
                    pl.len()
                    .over(
                        'ring_id'
                    )
                    .cast(pl.Float64)
                    .alias(
                        'ring_seed_count'
                    )
                )
                .with_columns(
                    # reset weight
                    (
                        1.0 
                        / pl.col('ring_seed_count')
                        / float(ring_count)
                    )
                    .alias(
                        'reset_weight'
                    )
                )
            )

            # reset vector
            reset_vector[
                seed_weights
                .get_column(
                    'node_id'
                )
                .to_numpy()
                .astype(
                    dtype = np.int64,
                    copy = False
                )
            ] = (
                seed_weights
                .get_column(
                    'reset_weight'
                )
                .to_numpy()
                .astype(
                    dtype = np.float64,
                    copy = False
                )
            )

        # reset sum
        reset_sum = float(
            reset_vector.sum()
        )

        # validate reset sum
        if not np.isclose(
            reset_sum,
            1.0,
            rtol = 1e-10,
            atol = 1e-12
        ):
            raise ValueError(
                'Personalization vector must sum to 1.0'
            )

        # store reset vector in the cache
        self._reset_vector = reset_vector

        return self._reset_vector

    ### public metadata and score methods
    @classmethod
    def get_ordinary_pagerank_columns(cls) -> tuple[str, ...]:
        """ Return ordinary PageRank score columns """
        return (
            *cls.ORDINARY_PAGERANK_CONFIGURATIONS.keys(),
            *cls.BIDIRECTIONAL_ORDINARY_PAGERANK_COLUMNS.keys()
        )

    @classmethod
    def get_personalized_pagerank_columns(cls) -> tuple[str, ...]:
        """ Return personalized PageRank score columns """
        return (
            *cls.PERSONALIZED_PAGERANK_CONFIGURATIONS.keys(),
            *cls.BIDIRECTIONAL_PERSONALIZED_PAGERANK_COLUMNS.keys()
        )

    @classmethod
    def get_score_columns(cls) -> tuple[str, ...]:
        """ Return reference, ordinary, and personalized ranking columns """
        return (
            *cls.REFERENCE_SCORE_COLUMNS,
            *cls.get_ordinary_pagerank_columns(),
            *cls.get_personalized_pagerank_columns()
        )
        
