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
            'log_count_weight'
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
            'log_count_weight'
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

    def _get_directional_reachability_mask(
            self,
            direction: str
    ) -> np.ndarray:
        """ Mark vertices reachable from at least one pooled seed """
        # vaidate if direction is either forward or reverse
        if not direction in ('forward', 'reverse'):
            raise ValueError(
                "direction must be either 'forward' or 'reverse'"
            )

        # check if self._reachability_cache is already constructed for a given direction
        if direction in self._reachability_cache:
            return self._reachability_cache[direction]

        # build igraph
        graph = (
            self.build_igraph(
                force_rebuild = False
            )
            if direction == 'forward'
            else self._get_reverse_graph()
        )

        # traversal graph
        traversal_graph = graph.copy()

        # super source node id
        super_source_node_id = traversal_graph.vcount()

        # add vertex and edges to the travelsal graph
        traversal_graph.add_vertex()
        traversal_graph.add_edges(
            [
                (super_source_node_id, int(seed_node_id))
                for seed_node_id in self._get_seed_node_ids()
            ]
        )

        # reachable node ids
        reachable_node_ids = np.asarray(
            traversal_graph.subcomponent(
                super_source_node_id,
                mode = 'out'
            ),
            dtype = np.int64
        )

        # exclude reachable nodes that do not fall within graph vertex range
        reachable_node_ids = (
            reachable_node_ids[
                reachable_node_ids < graph.vcount()
            ]
        )

        # construct reachable nodes array
        reachable_mask = np.zeros(
            graph.vcount(),
            dtype = bool
        )

        reachable_mask[reachable_node_ids] = True

        # assign reachable nodes array to self._reachability_cache for a given direction
        self._reachability_cache[direction] = reachable_mask

        return self._reachability_cache[direction]

    def _get_weak_reachability_mask(self) -> np.ndarray:
        """ Mark vertices sharing a weak component with any pooled seed """
        cache_key = 'weak'

        # validate if self._reachability_cache is not already built for cache_key
        if cache_key not in self._reachability_cache:
            # build graph
            graph = self.build_igraph(
                force_rebuild = False
            )

            # membership for weakly connected components
            membership = np.asarray(
                graph.connected_components(
                    mode = 'weak'
                ).membership,
                dtype = np.int64
            )

            # seed component ids
            seed_component_ids = np.unique(
                membership[self._get_seed_node_ids()]
            )

            # assign weak connected components to self._reachability_cache
            self._reachability_cache[cache_key] = np.isin(
                membership,
                seed_component_ids
            )

        return self._reachability_cache[cache_key]

    def _compute_pagerank_scores(
            self,
            direction: str,
            weight_attribute: str | None,
            personalized: bool
    ) -> np.ndarray:
        """ Compute and validate one ordinary or personalized score vector """
        # build igraph
        graph = (
            self.build_igraph()
            if direction == 'forward'
            else self._get_reverse_graph()
        )

        # compute PageRank scores
        if personalized:
            score_values = (
                graph.personalized_pagerank(
                    directed = True,
                    damping = self.damping,
                    reset = self._build_reset_vector().tolist(),
                    weights = weight_attribute,
                    implementation = 'prpack'
                )
            )
        else:
            score_values = (
                graph.pagerank(
                    directed = True,
                    damping = self.damping,
                    weights = weight_attribute,
                    implementation = 'prpack'
                )
            )

        scores = np.asarray(
            score_values,
            dtype = np.float64
        )

        # validate score vector shape is consistent with graph vertex
        if scores.shape[0] != graph.vcount():
            raise RuntimeError(
                'igraph returned an unexpected PageRank score shape'
            )

        # validate if scores contain all finite values
        if not np.isfinite(scores).all():
            raise RuntimeError(
                'PageRank returned non-finite scores'
            )

        # validate scores are non-negative
        if np.any(scores < 0.0):
            raise RuntimeError(
                'PageRank returned negative scores'
            )

        # validate sum of scores should be 1.0
        if not np.isclose(
            float(scores.sum()),
            1.0,
            rtol = 1e-8,
            atol = 1e-10
        ):
            raise RuntimeError(
                'PageRank scores must sum to 1.0'
            )
        
        return scores

    def _attach_target_labels(
            self,
            score_table: pl.DataFrame            
    ) -> pl.DataFrame:
        """ Attach held-out targets after scoring and exclude known seeds """
        # target table
        target_table = (
            self._collect_target_table()
            .select(
                [
                    'node_id',
                    'ring_id',
                    'target_order'
                ]
            )
            .with_columns(
                # ring target label
                pl.lit(True)
                .alias(
                    'is_ring_target'
                )
            )
        )

        return (
            score_table
            .join(
                target_table,
                on = 'node_id',
                how = 'left'
            )
            .with_columns(
                # ring target label
                pl.col('is_ring_target')
                .fill_null(False)
            )
            .filter(
                ~pl.col('is_seed')
            )
        )
    

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

    def build_input_summary(self) -> pl.DataFrame:
        """ Return one-row population and graph statistics """
        # summary table
        summaries = pl.collect_all(
            lazy_frames = [
                # node table
                self.nodes
                .select(
                    # node count
                    pl.len()
                    .alias(
                        'node_count'
                    )
                ),
                # edge table
                self.edges
                .select(
                    # edge count
                    pl.len()
                    .alias(
                        'edge_count'
                    )
                ),
                # ring table
                self.rings
                .select(
                    # ring count
                    pl.len()
                    .alias(
                        'ring_count'
                    )
                )
            ],
            engine = 'streaming'
        )

        # seed count
        seed_count = (
            self._collect_seed_table()
            .height
        )

        # target count
        target_count = (
            self._collect_target_table()
            .height
        )

        # node count
        node_count = int(
            summaries[0]
            .get_column(
                'node_count'
            )
            .item()
        )

        # candidate count
        candidate_count = node_count - seed_count

        return pl.DataFrame(
            {
                'node_count': [node_count],
                'edge_count': [
                    int(
                        summaries[1]
                        .get_column(
                            'edge_count'
                        )
                        .item()
                    )
                ],
                'ring_count': [
                    int(
                        summaries[2]
                        .get_column(
                            'ring_count'
                        )
                        .item()
                    )
                ],
                'seed_count': [seed_count],
                'candidate_count': [candidate_count],
                'target_count': [target_count],
                'target_rate': [
                    target_count / candidate_count
                ],
                'damping': [self.damping],
                'personalization_scheme': [self.personalization_scheme]
            }
        )

    def build_score_table(
            self,
            force_recompute: bool = False
    ) -> pl.DataFrame:
        """ Compute graph scores without attaching targets or ring identifiers """
        # return self._score table if it is already stored in the cache and force_recompute is False
        if self._score_table is not None and not force_recompute:
            return self._score_table

        # construct score table
        score_values: dict[str, np.ndarray] = {}

        # 1 - Ordinary PageRank 
        for score_name, configuration in self.ORDINARY_PAGERANK_CONFIGURATIONS.items():
            # configuration
            direction, weight_attribute = configuration

            # compute scores
            score_values[score_name] = self._compute_pagerank_scores(
                direction = direction,
                weight_attribute = weight_attribute,
                personalized = False
            )

        # 2 - Bidirectional Ordinary PageRank
        for score_name, source_columns in self.BIDIRECTIONAL_ORDINARY_PAGERANK_COLUMNS.items():
            # source
            forward_column, reverse_column = source_columns

            # compute scores
            score_values[score_name] = (
                (score_values[forward_column] + score_values[reverse_column]) / 2.0
            )

        # 3 - Personalized PageRank
        for score_name, configuration in self.PERSONALIZED_PAGERANK_CONFIGURATIONS.items():
            # configuration
            direction, weight_attribute = configuration

            # compute scores
            score_values[score_name] = self._compute_pagerank_scores(
                direction = direction,
                weight_attribute = weight_attribute,
                personalized = True
            )

        # 4 - Bidirectional Personalized PageRank
        for score_name, source_columns in self.BIDIRECTIONAL_PERSONALIZED_PAGERANK_COLUMNS.items():
            # source
            forward_column, reverse_column = source_columns

            # compute scores
            score_values[score_name] = (
                (score_values[forward_column] + score_values[reverse_column]) / 2.0
            )

        # validate score values sum to 1.0
        for score_name, scores in score_values.items():
            if not np.isclose(
                float(scores.sum()),
                1.0,
                rtol = 1e-8,
                atol = 1e-10
            ):
                raise RuntimeError(
                    f'{score_name} scores must sum to 1.0'
                )

        # score series
        score_series = [
            pl.Series(
                name = score_name,
                values = score_values[score_name],
                dtype = pl.Float64                
            )
            for score_name in (
                *self.get_ordinary_pagerank_columns(),
                *self.get_personalized_pagerank_columns()
            )
        ]

        # score table
        self._score_table = (
            self._build_base_node_table()
            .with_columns(
                [
                    # in seed weak component label
                    pl.Series(
                        name = 'is_in_seed_weak_component',
                        values = self._get_weak_reachability_mask(),
                        dtype = pl.Boolean
                    ),
                    # forward reachable from seed label
                    pl.Series(
                        name = 'is_forward_reachable_from_seed',
                        values = self._get_directional_reachability_mask(
                            direction = 'forward'
                        ),
                        dtype = pl.Boolean
                    ),
                    # reverse reachability from seed label
                    pl.Series(
                        name = 'is_reverse_reachable_from_seed',
                        values = self._get_directional_reachability_mask(
                            direction = 'reverse'
                        ),
                        dtype = pl.Boolean
                    ),
                    # PageRank scores
                    *score_series
                ]
            )
        )

        # clear candidate score table cache
        self._candidate_score_table = None

        return self._score_table

    def build_candidate_score_table(
            self,
            force_recompute: bool = False
    ) -> pl.DataFrame:
        """ Attach target labels after scoring and return non-seed candidates """
        # compute self._candidate_score_table if it is not already in the cache or force_recompute is True
        if self._candidate_score_table is None or force_recompute:
            self._candidate_score_table = self._attach_target_labels(
                score_table = self.build_score_table(
                    force_recompute = force_recompute
                )
            )

        return self._candidate_score_table

    def get_score_mass_summary(
            self,
            score_columns: Sequence[str] | None = None
    ) -> pl.DataFrame:
        """ Summarize how each score distributes mass across populations """
        # validate score columns
        selected_columns = self._validate_score_columns(
            score_columns = (
                score_columns
                if score_columns is not None
                else self.get_score_columns()
            )
        )

        # candidate table
        candidate_table = self.build_candidate_score_table(
            force_recompute = False
        )

        # full score table
        full_score_table = self.build_score_table(
            force_recompute = False
        )

        # target mask
        target_mask = np.zeros(
            full_score_table.height,
            dtype = bool
        )

        target_mask[self._get_target_node_ids()] = True

        # labeled full score table
        labeled_full_table = (
            full_score_table
            .with_columns(
                pl.Series(
                    name = 'is_ring_target',
                    values = target_mask,
                    dtype = pl.Boolean
                )
            )
        )

        # construct score mass summary
        summary_rows: list[dict[str, str | float]] = []

        for score_column in selected_columns:
            # total mass
            total_mass = float(
                labeled_full_table
                .get_column(
                    score_column
                )
                .cast(pl.Float64)
                .sum()
            )

            # seed mass
            seed_mass = float(
                labeled_full_table
                .filter(
                    pl.col('is_seed')
                )
                .get_column(
                    score_column
                )
                .cast(pl.Float64)
                .sum()
            )

            # target mass
            target_mass = float(
                labeled_full_table
                .filter(
                    'is_ring_target'
                )
                .get_column(
                    score_column
                )
                .cast(pl.Float64)
                .sum()
            )

            # candidate mass
            candidate_mass = float(
                candidate_table
                .get_column(
                    score_column
                )
                .cast(pl.Float64)
                .sum()
            )

            summary_rows.append(
                {
                    'score_name': score_column,
                    'total_score_mass': total_mass,
                    'seed_score_mass': seed_mass,
                    'candidate_score_mass': candidate_mass,
                    'target_score_mass': target_mass,
                    'target_share_of_candidate_mass': (
                        target_mass / candidate_mass
                        if candidate_mass > 0.0
                        else 0.0
                    )
                }
            )

        return pl.DataFrame(summary_rows)

    def get_candidate_score_summary(
            self,
            score_columns: Sequence[str] | None = None
    ) -> pl.DataFrame:
        """ Summarize candidate scores separately for targets and negatives """
        # validate score columns
        selected_columns = self._validate_score_columns(
            score_columns = (
                score_columns
                if score_columns is not None
                else self.get_score_columns()
            )
        )

        # candidate score table
        candidates = self.build_candidate_score_table(
            force_recompute = False
        )

        # populations
        populations = {
            'all_candidates': candidates,
            'ring_targets': candidates.filter(
                pl.col('is_ring_target')
            ),
            'non_targets': candidates.filter(
                ~pl.col('is_ring_target')
            )
        }

        # construct candidate score summary
        summary_rows: list[dict[str, str | int | float]] = []

        for score_column in selected_columns:
            for population_name, population in populations.items():
                # check whether provided population is empty 
                if population.is_empty():
                    summary_rows.append(
                        {
                            'score_name': score_column,
                            'population': population_name,
                            'account_count': 0,
                            'minimum': float('nan'),
                            'mean': float('nan'),
                            'median': float('nan'),
                            'p95': float('nan'),
                            'p99': float('nan'),
                            'maximum': float('nan'),
                            'zero_share': float('nan')
                        }
                    )

                    continue

                # population score values
                score_values = (
                    population
                    .get_column(
                        score_column
                    )
                    .cast(pl.Float64)
                    .to_numpy()
                    .astype(
                        dtype = np.float64,
                        copy = False
                    )
                )

                summary_rows.append(
                    {
                        'score_name': score_column,
                        'population': population_name,
                        'account_count': population.height,
                        'minimum': float(
                            np.min(score_values)
                        ),
                        'mean': float(
                            np.mean(score_values)
                        ),
                        'median': float(
                            np.median(score_values)
                        ),
                        'p95': float(
                            np.quantile(
                                score_values,
                                q = 0.95
                            )
                        ),
                        'p99': float(
                            np.quantile(
                                score_values,
                                q = 0.99
                            )
                        ),
                        'maximum': float(
                            np.max(score_values)
                        ),
                        'zero_share': float(
                            np.mean(score_values == 0)
                        )
                    }
                )

        return pl.DataFrame(summary_rows)

    ### public evaluation methods
    def evaluate_global_rankings(
            self,
            k_values: Sequence[int] = (
                100,
                500,
                1_000,
                5_000,
                10_000
            ),
            score_columns: Sequence[str] | None = None
    ) -> pl.DataFrame:
        """
        Evaluate candidate rankings using binary held-out target labels

        Ties are resolved deterministically by ascending node IDs
        """
        # validate score columns
        selected_columns = self._validate_score_columns(
            score_columns = (
                score_columns
                if score_columns is not None
                else self.get_score_columns()
            )
        )

        # validate k_values
        selected_k_values = self._validate_k_values(
            k_values = k_values
        )

        # candidates score table
        candidates = self.build_candidate_score_table(
            force_recompute = False 
        )

        # validate if candidates score table is empty
        if candidates.is_empty():
            raise ValueError(
                'Global ranking evaluation requires candidates'
            )

        # candidates count
        candidate_count = candidates.height

        # target count
        target_count = int(
            candidates
            .get_column(
                'is_ring_target'
            )
            .sum()
        )

        # validate if target count is null
        if target_count == 0:
            raise ValueError(
                'Global ranking evaluation requires held-out targets'
            )

        # target rate
        target_rate = (
            target_count / candidate_count
        )

        # global evaluation scores
        evaluation_rows: list[dict[str, str | int | float]] = []

        for score_column in selected_columns:
            # ranked labels
            ranked_labels = (
                candidates
                .sort(
                    [
                        score_column,
                        'node_id'
                    ],
                    descending = [
                        True,
                        False
                    ]
                )
                .get_column(
                    'is_ring_target'
                )
                .to_numpy()
                .astype(
                    dtype = np.int8,
                    copy = False
                )
            )

            for requested_k in selected_k_values:
                # effective k value
                effective_k = min(
                    requested_k,
                    candidate_count
                )

                # top labels
                top_labels = ranked_labels[:effective_k]

                # hit count
                hit_count = int(
                    top_labels.sum()
                )

                # evaluation metrics
                precision_at_k = (
                    hit_count / effective_k
                )

                recall_at_k = (
                    hit_count / target_count
                )

                lift_at_k = (
                    precision_at_k / target_rate
                )

                discounts = (
                    1.0
                    /
                    np.log2(
                        np.arange(
                            2,
                            effective_k + 2
                        )
                    )
                )

                dcg_at_k = float(
                    np.dot(
                        top_labels,
                        discounts
                    )
                )

                ideal_hit_count = min(
                    target_count,
                    effective_k
                )

                ideal_dcg_at_k = float(
                    discounts[:ideal_hit_count]
                    .sum()
                )

                ndcg_at_k = (
                    dcg_at_k / ideal_dcg_at_k
                    if ideal_dcg_at_k > 0.0
                    else 0.0
                )

                evaluation_rows.append(
                    {
                        'score_name': score_column,
                        'requested_k': requested_k,
                        'effective_k': effective_k,
                        'candidate_count': candidate_count,
                        'target_count': target_count,
                        'target_rate': target_rate,
                        'hit_count': hit_count,
                        'precision_at_k': precision_at_k,
                        'recall_at_k': recall_at_k,
                        'lift_at_k': lift_at_k,
                        'ndcg_at_k': ndcg_at_k
                    }
                )

        return pl.DataFrame(evaluation_rows)

    def evaluate_target_ranks(
            self,
            score_columns: Sequence[str] | None = None
    ) -> pl.DataFrame:
        """ Evaluate complete-ranking target positions, average precision, and mean reciprocal rank """
        # validate score columns
        selected_columns = self._validate_score_columns(
            score_columns = (
                score_columns
                if score_columns is not None
                else self.get_score_columns()
            )
        )

        # candidates score table
        candidates = self.build_candidate_score_table()

        # candidate count
        candidate_count = candidates.height

        # target count
        target_count = int(
            candidates
            .get_column(
                'is_ring_target'
            )
            .sum()
        )

        # target rank summary
        result_rows: list[dict[str, str | int | float]] = []

        for score_column in selected_columns:
            # ranked labels
            ranked_labels = (
                candidates
                .sort(
                    [
                        score_column,
                        'node_id'
                    ],
                    descending = [
                        True,
                        False
                    ]
                )
                .get_column(
                    'is_ring_target'
                )
                .to_numpy()
                .astype(
                    bool,
                    copy = False
                )
            )

            # target ranks
            target_ranks = np.flatnonzero(ranked_labels) + 1

            # validate target_ranks size
            if target_ranks.size != target_count:
                raise RuntimeError(
                    'Target-rank extraction produced an unexpected count'
                )

            # evaluation metric
            precision_at_target_ranks = (
                np.arange(
                    1,
                    target_count + 1,
                    dtype = np.float64
                )
                /
                target_ranks
            )

            result_rows.append(
                {
                    'score_name': score_column,
                    'candidate_count': candidate_count,
                    'target_count': target_count,
                    'best_target_rank': int(
                        target_ranks.min()
                    ),
                    'median_target_rank': float(
                        np.median(target_ranks)
                    ),
                    'mean_target_rank': float(
                        target_ranks.mean()
                    ),
                    'p90_target_rank': float(
                        np.quantile(
                            target_ranks,
                            q = 0.90
                        )
                    ),
                    'worst_target_rank': int(
                        target_ranks.max()
                    ),
                    'median_target_rank_share': float(
                        np.median(target_ranks) / candidate_count
                    ),
                    'mean_reciprocal_rank': float(
                        1.0 / target_ranks.min()
                    ),
                    'average_precision': float(
                        precision_at_target_ranks.mean()
                    )
                }
            )
        
        return pl.DataFrame(result_rows)

    def build_ring_recovery_table(
            self,
            score_column: str,
            k_value: int
    ) -> pl.DataFrame:
        """ Return per-ring recovery statistics within a global top-k list """
        # validate score column
        selected_score = self._validate_score_columns(
            score_columns = score_column
        )[0]

        # validate k_value non-negativity
        self._validate_positive_integer(
            value = k_value,
            parameter_name = 'k_value'
        )

        # candidates score table
        candidates = self.build_candidate_score_table()

        # effective k
        effective_k = min(
            candidates.height,
            k_value
        )

        # ranked candidates
        ranked_candidates = (
            candidates
            .sort(
                [
                    selected_score,
                    'node_id'
                ],
                descending = [
                    True,
                    False
                ]
            )
            .with_row_index(
                name = 'global_rank',
                offset = 1
            )
        )

        # recovered targets
        recovered_targets = (
            ranked_candidates
            .head(
                effective_k
            )
            .filter(
                pl.col('is_ring_target')
            )
            .group_by(
                'ring_id'
            )
            .agg(
                [
                    # recovered target count
                    pl.len()
                    .cast(pl.UInt64)
                    .alias(
                        'recovered_target_count'
                    ),
                    # first recovered global rank
                    pl.col('global_rank')
                    .min()
                    .cast(pl.UInt64)
                    .alias(
                        'first_recovered_global_rank'
                    )
                ]
            )
        )

        # ring targets
        ring_targets = (
            self.rings
            .select(
                [
                    'ring_id',
                    'ring_size',
                    'seed_count',
                    'target_count'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

        return (
            ring_targets
            .join(
                recovered_targets,
                on = 'ring_id',
                how = 'left'
            )
            .with_columns(
                pl.col('recovered_target_count')
                .fill_null(0)
                .cast(pl.UInt64)
            )
            .with_columns(
                [
                    # ring target recall
                    (
                        pl.col('recovered_target_count') / pl.col('target_count')
                    )
                    .cast(pl.Float64)
                    .alias(
                        'ring_target_recall'
                    ),
                    # ring hit label
                    (
                        pl.col('recovered_target_count') > 0
                    )
                    .alias(
                        'is_ring_hit'
                    ),
                    # fully recovered ring label
                    (
                        pl.col('recovered_target_count') == pl.col('target_count')
                    )
                    .alias(
                        'is_fully_recovered_ring'
                    )
                ]
            )
            .with_columns(
                [
                    # score name
                    pl.lit(selected_score)
                    .alias(
                        'score_name'
                    ),
                    # requested k
                    pl.lit(k_value)
                    .alias(
                        'requested_k'
                    ),
                    # effective k
                    pl.lit(effective_k)
                    .alias(
                        'effective_k'
                    )
                ]
            )
            .select(
                [
                    'score_name',
                    'requested_k',
                    'effective_k',
                    'ring_id',
                    'ring_size',
                    'seed_count',
                    'target_count',
                    'recovered_target_count',
                    'ring_target_recall',
                    'is_ring_hit',
                    'is_fully_recovered_ring',
                    'first_recovered_global_rank'
                ]
            )
            .sort(
                [
                    'ring_target_recall',
                    'first_recovered_global_rank',
                    'ring_id'
                ],
                descending = [
                    True,
                    False,
                    False
                ],
                nulls_last = True
            )
        )

    def evaluate_ring_recovery(
            self,
            k_values: Sequence[int] = (
                100,
                00,
                1_000,
                5_000,
                10_000
            ),
            score_columns: Sequence[str] | None = None
    ) -> pl.DataFrame:
        """ Evaluate macro ring recovery within global candidate rankings """
        # validate score columns
        selected_columns = self._validate_score_columns(
            score_columns = (
                score_columns
                if score_columns is not None
                else self.get_score_columns()
            )
        )

        # validate k_values
        selected_k_values = self._validate_k_values(
            k_values = k_values
        )

        # candidate count
        candidate_count = (
            self.build_candidate_score_table()
            .height
        )

        # validate candidate count
        if candidate_count == 0:
            raise ValueError(
                'There are no candidates to rank'
            )

        # ring recovery evaluation summary
        summary_rows: list[dict[str, str | int | float]] = []

        for score_column in selected_columns:
            for k_value in selected_k_values:
                # recovery table
                recovery_table = self.build_ring_recovery_table(
                    score_column = score_column,
                    k_value = k_value
                )

                # effectve k
                effective_k = min(
                    candidate_count,
                    k_value
                )

                # ring count
                ring_count = recovery_table.height

                # ring hit count
                ring_hit_count = int(
                    recovery_table
                    .get_column(
                        'is_ring_hit'
                    )
                    .sum()
                )

                # fully recovered ring count
                fully_recovered_ring_count = int(
                    recovery_table
                    .get_column(
                        'is_fully_recovered_ring'
                    )
                    .sum()
                )

                # recovered target count
                recovered_target_count = int(
                    recovery_table
                    .get_column(
                        'recovered_target_count'
                    )
                    .sum()
                )

                # ring recall array
                ring_recall_values = np.asarray(
                    recovery_table
                    .get_column('ring_target_recall')
                    .to_numpy()
                    .astype(
                        dtype = np.float64,
                        copy = False
                    )
                )

                # first hit ranks
                first_hit_ranks = (
                    recovery_table
                    .get_column('first_recovered_global_rank')
                    .drop_nulls()
                    .to_numpy()
                    .astype(
                        dtype = np.float64,
                        copy = False
                    )
                )

                # mean & median ring target recall
                mean_ring_target_recall = float(
                    np.mean(
                        ring_recall_values
                    )
                )

                median_ring_target_recall = float(
                    np.median(
                        ring_recall_values
                    )
                )

                # mean first hit rank
                mean_first_hit_rank = (
                    float(
                        np.mean(
                            first_hit_ranks
                        )
                    )
                    if first_hit_ranks.size > 0
                    else float('nan')
                )

                summary_rows.append(
                    {
                        'score_name': score_column,
                        'requested_k': k_value,
                        'effective_k': effective_k,
                        'ring_count': ring_count,
                        'recovered_target_count': recovered_target_count,
                        'ring_hit_count': ring_hit_count,
                        'ring_hit_rate': (
                            ring_hit_count / ring_count
                        ),
                        'fully_recovered_ring_count': fully_recovered_ring_count,
                        'fully_recovered_ring_rate': (
                            fully_recovered_ring_count / ring_count
                        ),
                        'mean_ring_target_recall': mean_ring_target_recall,
                        'median_ring_target_recall': median_ring_target_recall,
                        'mean_first_hit_rank': mean_first_hit_rank
                    }
                )

        return pl.DataFrame(summary_rows)

    def get_top_ranked_accounts(
            self,
            score_column: str,
            top_n: int = 20
    ) -> pl.DataFrame:
        """ Return the highest-ranked non-seed candidate accounts """
        # validate score column
        selected_score = self._validate_score_columns(
            score_columns = score_column
        )[0]

        # validate top_n non-negativity
        self._validate_positive_integer(
            value = top_n,
            parameter_name = 'top_n' 
        )

        return (
            self.build_candidate_score_table()
            .sort(
                [
                    selected_score,
                    'node_id'
                ],
                descending = [
                    True,
                    False
                ]
            )
            .with_row_index(
                name = 'global_rank',
                offset = 1
            )
            .select(
                [
                    'global_rank',
                    'node_id',
                    'account',
                    selected_score,
                    'is_ring_target',
                    'ring_id',
                    'target_order',
                    'out_degree',
                    'in_degree',
                    'total_directed_degree',
                    'account_transaction_event_count',
                    'is_in_seed_weak_component',
                    'is_forward_reachable_from_seed',
                    'is_reverse_reachable_from_seed'
                ]
            )
            .head(
                n = top_n
            )
        )
          


    
