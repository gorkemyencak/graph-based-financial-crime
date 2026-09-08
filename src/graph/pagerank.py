from collections.abc import Sequence

import numpy as np
import polars as pl
import igraph as ig

from src.graph.eda import SAMLDGraphEDA

class SAMLDPageRankBaseline:
    """
    Build and evaluate validation-stage PageRank baselines

    Scores are computed exclusively from the historical graph and historical suspicious seeds. Future target labels are attached only
    when candidate rankings are evaluated

    Parameters:
        graph_eda:
            Initialized graph-EDA object for one persisted snapshot
        damping:
            PageRank damping factor. A value of 0.85 is the standard baseline    
    """
    # define reference score columns
    REFERENCE_SCORE_COLUMNS: tuple[str, ...] = (
        'out_degree',
        'in_degree',
        'total_directed_degree',
        'account_transaction_event_count'
    )

    # pagerank configurations
    PAGERANK_CONFIGURATIONS: dict[str, tuple[str, str | None]] = {
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

    # reachability metrics
    REACHABILITY_COLUMNS: dict[str, str | None] = {
        'all_candidates': None,
        'seed_weak_component': 'is_in_seed_weak_component',
        'forward_seed_reachable': 'is_forward_reachable_from_seed',
        'reverse_seed_reachable': 'is_reverse_reachable_from_seed'
    }

    def __init__(
            self,
            graph_eda: SAMLDGraphEDA,
            damping: float | int = 0.85
    ) -> None:
        # validate damping factor
        self._validate_damping(
            damping = damping
        )

        # attributes
        self.graph_eda = graph_eda
        self.damping = float(damping)

        # reuse the graph already created during graph EDA when it is cached
        self._graph = (
            self.graph_eda
            .build_igraph(
                force_rebuild = False
            )
        )

        # chech if igraph is directed
        if not self._graph.is_directed():
            raise ValueError(
                'PageRank baselines require a directed transaction graph'
            )

        # PageRank attributes
        self._reverse_graph: ig.Graph | None = None
        self._base_node_table: pl.DataFrame | None = None
        self._node_reachability_table: pl.DataFrame | None = None
        self._score_table: pl.DataFrame | None = None
        self._seed_node_ids: np.ndarray | None = None
        self._rankable_target_node_ids: np.ndarray | None = None
        self._reachability_cache: dict[str, np.ndarray] = {}

    ### private helper methods
    @staticmethod
    def _validate_damping(
        damping: float
    ) -> None:
        """ Validate the PageRank damping factor """
        # check if damping is either an integer or a float instance
        if not isinstance(damping, int | float):
            raise TypeError(
                'damping must be numeric'
            )

        # logical expression on damping factor
        if not 0.0 < float(damping) < 1.0:
            raise ValueError(
                'damping must be strictly between 0.0 and 1.0'
            )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        parameter_name: str
    ) -> None:
        """ Validate a strictly positive integer parameter """
        # check if value is an integer instance
        if not isinstance(value, int):
            raise TypeError(
                f'{parameter_name} must be an integer'
            )

        # non-negativity
        if value <= 0:
            raise ValueError(
                f'{parameter_name} must be greater than 0'
            )

    @classmethod
    def _validate_score_columns(
        cls,
        score_columns: Sequence[str]
    ) -> tuple[str, ...]:
        """ Validate and deduplicate requested score columns """
        # check if score columns are a string instance
        if isinstance(score_columns, str):
            requested_columns = (score_columns, )
        else:
            requested_columns = tuple(score_columns)

        # check if requesting columns is an empty object
        if not requested_columns:
            raise ValueError(
                'At least one score column must be requested'
            )

        # valid columns
        valid_columns = set(
            cls.get_score_columns()
        )

        # invalid columns
        invalid_columns = (
            set(requested_columns) - valid_columns
        )

        # check if invalid columns present, return sorted missing columns in the valid columns set
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

        return tuple(
            dict.fromkeys(
                requested_columns
            )
        )

    def _get_seed_node_ids(self) -> np.ndarray:
        """ Return unique historical seed node IDs """
        # check if seed node ids not in the cache, then construct
        if self._seed_node_ids is None:
            seed_node_ids = (
                self.graph_eda.seeds
                .select(
                    'node_id'
                )
                .unique()
                .sort(
                    'node_id'
                )
                .collect(
                    engine = 'streaming'
                )
                .get_column(
                    'node_id'
                )
                .to_numpy()
                .astype(
                    np.int64,
                    copy = False
                )
            )

            # check if seed node ids are empty on disk
            if seed_node_ids.size == 0:
                raise ValueError(
                    'Personalized reachability requires at least one seed'
                )

            # ensure that seed node ids fall within graph vertex range
            if(
                seed_node_ids.min() < 0
                or seed_node_ids.max() >= self._graph.vcount()
            ):
                raise ValueError(
                    'Seed node IDs fall outside the graph vertex range'
                )

            self._seed_node_ids = seed_node_ids

        return self._seed_node_ids

    def _get_rankable_target_node_ids(self) -> np.ndarray:
        """ Return validation targets that exist in the historical graph """
        # check if rankable target node ids in the cache
        if self._rankable_target_node_ids is None:
            target_node_ids = (
                self.graph_eda.targets
                .filter(
                    pl.col('is_rankable_new_suspicious_account')
                )
                .select(
                    'node_id'
                )
                .drop_nulls()
                .unique()
                .sort(
                    'node_id'
                )
                .collect(
                    engine = 'streaming'
                )
                .get_column(
                    'node_id'
                )
                .to_numpy()
                .astype(
                    np.int64,
                    copy = False
                )
            )

            # check if target node ids are empty on disk
            if target_node_ids.size == 0:
                raise ValueError(
                    'Personalized reachability requires at least one target'
                )
            
            # ensure that target node ids fall within graph vertex range
            if(
                target_node_ids.min() < 0
                or target_node_ids.max() >= self._graph.vcount()
            ):
                raise ValueError(
                    'Rankable target node IDs fall outside the graph range'
                )

            self._rankable_target_node_ids = target_node_ids

        return self._rankable_target_node_ids

    def _build_base_node_table(self) -> pl.DataFrame:
        """ Collect historical node fields in graph-vertex order """
        # return base node table if it is already present in the cache
        if self._base_node_table is not None:
            return self._base_node_table

        # node table
        node_table = (
            self.graph_eda.nodes
            .select(
                [
                    'node_id',
                    'account',
                    'out_degree',
                    'in_degree',
                    'total_directed_degree',
                    'account_transaction_event_count'
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
        node_count = (
            self._graph
            .vcount()
        )

        # node ids
        node_ids = (
            node_table
            .get_column(
                'node_id'
            )
            .to_numpy()
            .astype(
                np.int64,
                copy = False
            )
        )

        # logical expressions
        if node_table.height != node_count or not np.array_equal(
            node_ids,
            np.arange(
                node_count, 
                dtype = np.int64
            )
        ):
            raise ValueError(
                'Snapshot node IDs must be contiguous, zero-based and equal to igraph vertex indices'
            )

        # mark seed nodes
        seed_mask = np.zeros(
            node_count,
            dtype = bool
        )

        seed_mask[self._get_seed_node_ids()] = True

        # create base node table with seed mask
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

    def _get_reverse_graph(self) -> ig.Graph:
        """ Return a cached graph with every transaction edge reversed """
        # check if reverse graph is in the cache
        if self._reverse_graph is None:
            self._reverse_graph = self._graph.copy()

            # check if self._reverse_graph is an ig.Graph instance, then reverse every transaction edge of the cached graph
            if isinstance(self._reverse_graph, ig.Graph):
                self._reverse_graph.reverse_edges()

        return self._reverse_graph

    def _get_weak_seed_component_mask(self) -> np.ndarray:
        """ Mark nodes sharing an undirected component with any seed """
        # define cache key
        cache_key = 'weak'

        # check if cache key is already in self._reachability_cache, else construct
        if cache_key not in self._reachability_cache:
            # connected component membership
            membership = np.asarray(
                self._graph
                .connected_components(
                    mode = 'weak'
                )
                .membership,
                dtype = np.int64
            )

            # seed component ids
            seed_component_ids = np.unique(
                membership[self._get_seed_node_ids()]
            )

            # reachability
            self._reachability_cache[cache_key] = np.isin(
                membership,
                seed_component_ids
            )

        return self._reachability_cache[cache_key]

    def _get_directed_seed_reachability_mask(
            self,
            direction: str
    ) -> np.ndarray:
        """ Mark nodes reachable from any seed in one graph direction """
        # check direction is either forward or reverse
        if direction not in {'forward', 'reverse'}:
            raise ValueError(
                "direction must be either 'forward' or 'reverse'"
            )

        # check if direction is in reachability cache, return reachability
        if direction in self._reachability_cache:
            return self._reachability_cache[direction]

        # define source graph
        source_graph = (
            self._graph
            if direction == 'forward'
            else self._get_reverse_graph()
        )

        # a temporary super-source turns multi-seed reachability into one BFS
        traversal_graph = source_graph.copy()

        super_source_id = traversal_graph.vcount()

        traversal_graph.add_vertex()
        traversal_graph.add_edges(
            [
                (super_source_id, int(seed_node_id))
                for seed_node_id in self._get_seed_node_ids()
            ]
        )

        # define reachable node ids
        reachable_node_ids = np.asarray(
            traversal_graph.subcomponent(
                super_source_id,
                mode = 'out'
            ),
            dtype = np.int64
        )

        reachable_node_ids = reachable_node_ids[
            reachable_node_ids < self._graph.vcount()
        ]

        # mark reachable nodes
        reachable_mask = np.zeros(
            self._graph.vcount(),
            dtype = bool
        )
        reachable_mask[reachable_node_ids] = True

        self._reachability_cache[direction] = reachable_mask

        return self._reachability_cache[direction]

    def _compute_pagerank_scores(
            self,
            direction: str,
            weight_attribute: str | None
    ) -> np.ndarray:
        """ Compute one ordinary PageRank score vector """
        # define graph
        graph = (
            self._graph
            if direction == 'forward'
            else self._get_reverse_graph()
        )

        # PageRank scores
        scores = np.asarray(
            graph.pagerank(
                directed = True,
                damping = self.damping,
                weights = weight_attribute,
                implementation = 'prpack'
            ),
            dtype = np.float64
        )

        # check PageRank score vector shape
        expected_shape = (
            self._graph.vcount(), 
        )

        if scores.shape != expected_shape:
            raise RuntimeError(
                'igraph returned an unexpected PageRank score shape'
            )

        # check if score vector contains infinite scores
        if not np.isfinite(scores).all():
            raise RuntimeError(
                'PageRank returned non-finite scores'
            )

        return scores

    def _build_node_reachability_table(self) -> pl.DataFrame:
        """ Return historical nodes with seed and reachability flags """
        if self._node_reachability_table is None:
            # construct node reachability table
            self._node_reachability_table = (
                self._build_base_node_table()
                .with_columns(
                    [
                        pl.Series(
                            name = 'is_in_seed_weak_component',
                            values = self._get_weak_seed_component_mask(),
                            dtype = pl.Boolean
                        ),
                        pl.Series(
                            name = 'is_forward_reachable_from_seed',
                            values = self._get_directed_seed_reachability_mask(
                                direction = 'forward'
                            ),
                            dtype = pl.Boolean
                        ),
                        pl.Series(
                            name = 'is_reverse_reachable_from_seed',
                            values = self._get_directed_seed_reachability_mask(
                                direction = 'reverse'
                            ),
                            dtype = pl.Boolean
                        )
                    ]
                )
            )

        return self._node_reachability_table

    def _attach_validation_target_flag(
            self,
            node_table: pl.DataFrame
    ) -> pl.DataFrame:
        """ Attach future labels for evaluation and remove historical seeds """
        # mark rankable target nodes
        target_mask = np.zeros(
            self._graph.vcount(),
            dtype = bool
        )

        target_mask[self._get_rankable_target_node_ids()] = True

        return (
            node_table
            .with_columns(
                pl.Series(
                    name = 'is_rankable_target',
                    values = target_mask,
                    dtype = pl.Boolean
                )
            )
            .filter(
                ~pl.col('is_seed')
            )
        )
    

    ### public PageRank methods
    @classmethod
    def get_score_columns(cls) -> tuple[str, ...]:
        """ Return reference and PageRank score columns in evaluation order """
        return (
            *cls.REFERENCE_SCORE_COLUMNS,
            *cls.PAGERANK_CONFIGURATIONS.keys()
        )

    def build_score_table(
            self,
            force_recompute: bool = False
    ) -> pl.DataFrame:
        """  
        Return historical node metrics, reachability flags and PageRank scores
        
        No future target label is present in this table
        """
        # if score table is in cache and recompute is not needed
        if self._score_table is not None and not force_recompute:
            return self._score_table

        # compute score table
        score_series: list[pl.Series] = []

        for score_name, (direction, weight_attribute) in self.PAGERANK_CONFIGURATIONS.items():

            score_series.append(
                pl.Series(
                    name = score_name,
                    values = self._compute_pagerank_scores(
                        direction = direction,
                        weight_attribute = weight_attribute
                    ),
                    dtype = pl.Float64
                )
            )

        self._score_table = (
            self._build_node_reachability_table()
            .with_columns(
                score_series
            )
        )

        return self._score_table

    def build_candidate_score_table(
            self,
            force_recompute: bool = False
    ) -> pl.DataFrame:
        """ Return scores for non-seed candidates with validation labels """
        return self._attach_validation_target_flag(
            node_table = self.build_score_table(
                force_recompute = force_recompute
            )
        )

    def get_seed_reachability_summary(self) -> pl.DataFrame:
        """ Summarize candidate and target coverage from historical seeds """
        # candidate table
        candidates = self._attach_validation_target_flag(
            node_table = self._build_node_reachability_table()
        )

        # total candidate count
        total_candidate_count = candidates.height

        # total target count
        total_target_count = int(
            candidates
            .get_column(
                'is_rankable_target'
            )
            .sum()
        )

        # rankable target positive rate
        overall_positive_rate = (
            total_target_count / total_candidate_count
            if total_candidate_count > 0
            else 0.0
        )

        # seed reachability summary
        summary_rows: list[dict[str, str | int | float]] = []

        for reachability_mode, flag_column in self.REACHABILITY_COLUMNS.items():

            # reachable candidates
            reachable_candidates = (
                candidates
                if flag_column is None
                else candidates.filter(pl.col(flag_column))
            )

            # reachable candidates count
            reachable_candidate_count = reachable_candidates.height

            # reachable target count
            reachable_target_count = int(
                reachable_candidates
                .get_column(
                    'is_rankable_target'
                )
                .sum()
            )

            # reachable positive rate
            reachable_positive_rate = (
                reachable_target_count / reachable_candidate_count
                if reachable_candidate_count > 0
                else 0.0
            )

            summary_rows.append(
                {
                    'reachability_mode': reachability_mode,
                    'reachable_candidate_count': reachable_candidate_count,
                    'candidate_coverage': (
                        reachable_candidate_count / total_candidate_count
                        if total_candidate_count > 0
                        else 0.0
                    ),
                    'reachable_target_count': reachable_target_count,
                    'target_coverage': (
                        reachable_target_count / total_target_count
                        if total_target_count > 0
                        else 0.0
                    ),
                    'unreachable_target_count': (
                        total_target_count - reachable_target_count
                    ),
                    'reachable_positive_rate': reachable_positive_rate,
                    'positive_rate_lift': (
                        reachable_positive_rate / overall_positive_rate
                        if overall_positive_rate > 0
                        else 0.0
                    )
                }
            )

        return pl.DataFrame(summary_rows)

    def get_score_summary(
            self,
            score_columns: Sequence[str] | None = None
    ) -> pl.DataFrame:
        """ Summarize score distributions over non-seed candidates """
        # validate score columns
        selected_columns = self._validate_score_columns(
            score_columns = (
                score_columns
                if score_columns is not None
                else self.get_score_columns()
            )
        )

        # score table with non-seed candidates
        candidate_score_table = self.build_candidate_score_table()

        # check if candidate score table is empty
        if candidate_score_table.is_empty():
            raise ValueError(
                'Score summary requires at least one non-seed candidate'
            )

        # summary table
        summary_frames: list[pl.DataFrame] = []

        for score_column in selected_columns:

            score_expression = (
                pl.col(score_column)
                .cast(pl.Float64)
            )

            score_summary = (
                candidate_score_table
                .select(
                    [
                        # score name
                        pl.lit(score_column)
                        .alias(
                            'score_name'
                        ),
                        # minimum
                        score_expression
                        .min()
                        .alias(
                            'minimum'
                        ),
                        # mean
                        score_expression
                        .mean()
                        .alias(
                            'mean'
                        ),
                        # median
                        score_expression
                        .median()
                        .alias(
                            'median'
                        ),
                        # p95
                        score_expression
                        .quantile(0.95)
                        .alias(
                            'p95'
                        ),
                        # p99
                        score_expression
                        .quantile(0.99)
                        .alias(
                            'p99'
                        ),
                        # maximum
                        score_expression
                        .max()
                        .alias(
                            'maximum'
                        ),
                        # zero share
                        (
                            score_expression == 0.0
                        )
                        .mean()
                        .alias(
                            'zero_share'
                        )
                    ]
                )
            )

            summary_frames.append(
                score_summary
            )

        return (
            pl.concat(
                summary_frames,
                how = 'vertical'
            )
        )

    def evaluate_scores(
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
        Evaluate reference and PageRank rankings on validation candidates

        Ties are resolved deterministically by ascending node ID
        """
        # validate score columns
        selected_columns = self._validate_score_columns(
            score_columns = (
                score_columns
                if score_columns is not None
                else self.get_score_columns()
            )
        )

        # selected k values
        selected_k_values = tuple(
            sorted(
                set(k_values)
            )
        )

        # check if k_values empty
        if not selected_k_values:
            raise ValueError(
                'At least one k value must be requested'
            )

        # validate non-negativity
        for k_value in selected_k_values:
            self._validate_positive_integer(
                value = k_value,
                parameter_name = 'k_value'
            )

        # score table with non-seed candidates
        candidate_score_table = self.build_candidate_score_table()

        # check if candidate score table is empty
        if candidate_score_table.is_empty():
            raise ValueError(
                'Score summary requires at least one non-seed candidate'
            )

        # candidate score table count
        candidate_count = candidate_score_table.height

        # rankable target count
        positive_count = int(
            candidate_score_table
            .get_column(
                'is_rankable_target'
            )
            .sum()
        )

        # check rankable target count is non-empty
        if positive_count == 0:
            raise ValueError(
                'Candidate evaluation requires at least one rankable target'
            )

        # rankable target rate
        positive_rate = positive_count / candidate_count

        # evaluation summary
        evaluation_rows: list[dict[str, str | int | float]] = []

        for score_column in selected_columns:
            # ranked labels
            ranked_labels = (
                candidate_score_table
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
                    'is_rankable_target'
                )
                .to_numpy()
                .astype(
                    np.int8,
                    copy = False
                )
            )

            for requested_k in selected_k_values:
                # effective_k
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
                    hit_count / positive_count
                )

                lift_at_k = (
                    precision_at_k / positive_rate
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
                    positive_count,
                    effective_k
                )

                ideal_dcg_at_k = float(
                    discounts[:ideal_hit_count]
                    .sum()
                )

                ndcg_at_k = (
                    dcg_at_k / ideal_dcg_at_k
                    if ideal_dcg_at_k > 0
                    else 0.0
                )

                evaluation_rows.append(
                    {
                        'score_name': score_column,
                        'requested_k': requested_k,
                        'effective_k': effective_k,
                        'hit_count': hit_count,
                        'precision_at_k': precision_at_k,
                        'recall_at_k': recall_at_k,
                        'lift_at_k': lift_at_k,
                        'ndcg_at_k': ndcg_at_k
                    }
                )

        return pl.DataFrame(evaluation_rows)

    def get_top_ranked_accounts(
            self,
            score_column: str,
            top_n: int = 20
    ) -> pl.DataFrame:
        """ Return the highest-ranked validation candidates for one score """
        # validate score columns
        selected_score = self._validate_score_columns(
            score_columns = score_column
        )[0]

        # validate parameter non-negativity
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
            .select(
                [
                    'node_id',
                    'account',
                    selected_score,
                    'is_rankable_target',
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
                top_n
            )
        )