import polars as pl

from pathlib import Path
from collections.abc import Sequence

class SAMLDCandidateFeatureBuilder:
    """
    Build label-free account features for the SAML-D ring experiment

    The persisted feature table contains observation-graph information only. Held-out target labels and ring
    identifiers are attached only inside the explicit evaluation methods and are never written by write_feature_table()

    The feature layer combines:
        - node activity and degree features already stored in the score table
        - ordinary and personalized PageRank scores
        - directional transaction volume
        - direct seed-neighbor exposure
        - deterministic forward/reverse ranks
        - direction-aware maximum-score and reciprocal-rank fusion baselines
    """
    EXPERIMENT_FILE_NAMES: dict[str, str] = {
        'nodes': 'nodes.parquet',
        'edges': 'edges.parquet',
        'seeds': 'seeds.parquet',
        'targets': 'targets.parquet'
    }

    REQUIRED_COLUMNS: dict[str, set[str]] = {
        'nodes': {
            'node_id',
            'account',
            'out_degree',
            'in_degree',
            'total_directed_degree',
            'account_transaction_event_count'
        },
        'edges': {
            'source_node_id',
            'target_node_id',
            'count_weight',
            'log_count_weight'
        },
        'seeds': {
            'ring_id',
            'node_id',
            'account'
        },
        'targets': {
            'ring_id',
            'node_id',
            'account',
            'target_order'
        },
        'scores': {
            'node_id',
            'account',
            'is_seed',
            'out_degree',
            'in_degree',
            'total_directed_degree',
            'account_transaction_event_count',
            'is_in_seed_weak_component',
            'is_forward_reachable_from_seed',
            'is_reverse_reachable_from_seed',
            'personalized_pagerank_unweighted',
            'reverse_personalized_pagerank_unweighted',
            'bidirectional_personalized_pagerank_unweighted'
        }
    }

    FORBIDDEN_FEATURE_COLUMNS: set[str] = {
        'ring_id',
        'target_order',
        'is_ring_target',
        'is_laundering',
        'laundering_type'
    }

    FORWARD_PPR_COLUMN = 'personalized_pagerank_unweighted'

    REVERSE_PPR_COLUMN = 'reverse_personalized_pagerank_unweighted'

    BLENDED_PPR_COLUMN = 'bidirectional_personalized_pagerank_unweighted'

    MAX_PPR_COLUMN =  'directional_max_personalized_pagerank_unweighted'

    RRF_COLUMN = 'personalized_pagerank_reciprocal_rank_fusion'

    def __init__(
            self,
            experiment_dir: Path | str,
            score_path: Path | str | None = None,
            rrf_constant: float | int = 60.0
    ) -> None:
        # validate rrf constant
        self._validate_rrf_constant(
            rrf_constant = rrf_constant
        )

        # attributes
        self.experiment_dir = Path(experiment_dir)
        self.score_path = (
            Path(score_path)
            if score_path is not None
            else self.experiment_dir / 'personalized_pagerank_scores.parquet'
        )
        self.rrf_constant = float(rrf_constant)
        self.files = {
            name: self.experiment_dir / file_name
            for name, file_name in self.EXPERIMENT_FILE_NAMES.items()
        }
        self.files['scores'] = self.score_path

        # validate input files and schemas
        self._validate_input_files()
        self._validate_input_schemas()

        # store experiment files in the cache
        self.nodes = pl.scan_parquet(
            source = self.files['nodes']
        )
        self.edges = pl.scan_parquet(
            source = self.files['edges']
        )
        self.seeds = pl.scan_parquet(
            source = self.files['seeds']
        )
        self.targets = pl.scan_parquet(
            source = self.files['targets']
        )
        self.scores = pl.scan_parquet(
            source = self.files['scores']
        )

        # private class attributes
        self._edge_feature_table: pl.DataFrame | None = None
        self._feature_table: pl.DataFrame | None = None

    # private validation methods
    @staticmethod
    def _validate_rrf_constant(rrf_constant: float | int) -> None:
        """ Validate the reciprocal rank-fusion smoothing constant """
        # validate rrf_constant instance type
        if not isinstance(rrf_constant, int | float):
            raise TypeError(
                'rrf_constant must be either int or float'
            )

        # validate non-negative rrf_constant
        if float(rrf_constant) < 0.0:
            raise ValueError(
                'rrf_constant must be strictly positive'
            )

    def _validate_input_files(self) -> None:
        """ Validate that every required Parquet input exists and is non-empty """
        for table_name, path in self.files.items():
            # validate if file exists
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {table_name} Parquet file: {path}"
                )

            # validate if file is non-empty
            if path.stat().st_size == 0:
                raise ValueError(
                    f'The {table_name} Parquet file is empty: {path}'
                )

    def _validate_input_schemas(self) -> None:
        """ Validate the minimum schema required from each input table """
        for table_name, required_columns in self.REQUIRED_COLUMNS.items():
            # extract actual columns from Parquet file
            actual_columns = set(
                pl.scan_parquet(
                    source = self.files[table_name]
                )
                .collect_schema()
                .names()
            )

            # missing columns
            missing_columns = required_columns - actual_columns

            # validate whether missing columns present
            if missing_columns:
                sorted_missing_columns = ', '.join(
                    sorted(missing_columns)
                )

                raise ValueError(
                    f'{table_name!r} is missing required columns: {sorted_missing_columns}'
                )

        # validate leaked columns present in the score columns
        score_columns = set(
            pl.scan_parquet(
                source = self.files['scores']
            )
            .collect_schema()
            .names()
        )

        leaked_columns = self.FORBIDDEN_FEATURE_COLUMNS & score_columns

        if leaked_columns:
            sorted_leaked_columns = ', '.join(
                sorted(leaked_columns)
            )

            raise ValueError(
                f'The persisted score table contains evaluation labels: {sorted_leaked_columns}'
            )

    @staticmethod
    def _rank_candidates(
        feature_table: pl.DataFrame,
        score_column: str,
        rank_column: str
    ) -> pl.DataFrame:
        """ Create a deterministic descending rank with node_id tie-breaking """
        return (
            feature_table
            .select(
                [
                    'node_id',
                    score_column
                ]
            )
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
            .with_row_index(
                name = rank_column,
                offset = 1
            )
            .select(
                [
                    'node_id',
                    rank_column
                ]
            )
        )

    def _build_edge_feature_table(self) -> pl.DataFrame:
        """ Aggregate directional activity and direct exposure to known seeds """
        # return if edge feature table is already stored in the cache
        if self._edge_feature_table is not None:
            return self._edge_feature_table

        # unique seed ids
        seed_ids = (
            self.seeds
            .select(
                pl.col('node_id')
                .alias(
                    'seed_node_id'
                )
            )
            .unique()
        )

        # outgoing activity table
        outgoing_activity = (
            self.edges
            .group_by(
                'source_node_id'
            )
            .agg(
                [
                    # outgoing transaction count
                    pl.col('count_weight')
                    .sum()
                    .alias(
                        'outgoing_transaction_count'
                    ),
                    # outgoing log count weight
                    pl.col('log_count_weight')
                    .sum()
                    .alias(
                        'outgoing_log_count_weight'
                    )
                ]
            )
            .rename(
                {
                    'source_node_id': 'node_id'
                }
            )
        )

        # incoming activity table
        incoming_activity = (
            self.edges
            .group_by(
                'target_node_id'
            )
            .agg(
                [
                    # incoming transaction count
                    pl.col('count_weight')
                    .sum()
                    .alias(
                        'incoming_transaction_count'
                    ),
                    # incoming log count weight
                    pl.col('log_count_weight')
                    .sum()
                    .alias(
                        'incoming_log_count_weight'
                    )
                ]
            )
            .rename(
                {
                    'target_node_id': 'node_id'
                }
            )
        )

        # forward seed exposure
        forward_seed_exposure = (
            self.edges
            .join(
                seed_ids,
                left_on = 'source_node_id',
                right_on = 'seed_node_id',
                how = 'inner'
            )
            .group_by(
                'target_node_id'
            )
            .agg(
                [
                    # forward seed neighbor count
                    pl.col('source_node_id')
                    .n_unique()
                    .alias(
                        'forward_seed_neighbor_count'
                    ),
                    # forward seed transaction count
                    pl.col('count_weight')
                    .sum()
                    .alias(
                        'forward_seed_transaction_count'
                    ),
                    # forward seed log count weight
                    pl.col('log_count_weight')
                    .sum()
                    .alias(
                        'forward_seed_log_count_weight'
                    )
                ]
            )
            .rename(
                {
                    'target_node_id': 'node_id'
                }
            )
        )

        # reverse seed exposure
        reverse_seed_exposure = (
            self.edges
            .join(
                seed_ids,
                left_on = 'target_node_id',
                right_on = 'seed_node_id',
                how = 'inner'
            )
            .group_by(
                'source_node_id'
            )
            .agg(
                [
                    # reverse seed neighbor count
                    pl.col('target_node_id')
                    .n_unique()
                    .alias(
                        'reverse_seed_neighbor_count'
                    ),
                    # reverse seed transaction count
                    pl.col('count_weight')
                    .sum()
                    .alias(
                        'reverse_seed_transaction_count'
                    ),
                    # reverse seed log count weight
                    pl.col('log_count_weight')
                    .sum()
                    .alias(
                        'reverse_seed_log_count_weight'
                    )
                ]
            )
            .rename(
                {
                    'source_node_id': 'node_id'
                }
            )
        )

        # collect all tables
        collected_tables = pl.collect_all(
            lazy_frames = [
                outgoing_activity,
                incoming_activity,
                forward_seed_exposure,
                reverse_seed_exposure
            ],
            engine = 'streaming'
        )

        edge_features = collected_tables[0]

        for table in collected_tables[1:]:
            edge_features = (
                edge_features
                .join(
                    table,
                    on = 'node_id',
                    how = 'full',
                    coalesce = True
                )
            )

        self._edge_feature_table = edge_features

        return self._edge_feature_table

    @classmethod
    def _validate_feature_table(
        cls,
        feature_table: pl.DataFrame
    ) -> None:
        """ Validate the label-free candidate feature table before persistence """
        # validate if feature table is non-empty
        if feature_table.is_empty():
            raise ValueError(
                'The candidate feature table is empty'
            )

        # actual columns extracted from feature_table
        actual_columns = set(
            feature_table.columns
        )

        # expected columns
        required_columns = {
            'node_id',
            'account',
            cls.FORWARD_PPR_COLUMN,
            cls.REVERSE_PPR_COLUMN,
            cls.BLENDED_PPR_COLUMN,
            cls.MAX_PPR_COLUMN,
            cls.RRF_COLUMN
        }

        # missing columns
        missing_columns = required_columns - actual_columns

        # validate missing columns present
        if missing_columns:
            sorted_missing_columns = ', '.join(
                sorted(missing_columns)
            )

            raise ValueError(
                f'Candidate feature table has missing columns: {sorted_missing_columns}'
            )

        # validate leaked columns
        leaked_columns = cls.FORBIDDEN_FEATURE_COLUMNS & actual_columns

        if leaked_columns:
            sorted_leaked_columns = ', '.join(
                sorted(leaked_columns)
            )

            raise ValueError(
                f'Candidate feature table contains evaluation labels: {sorted_leaked_columns}'
            )

        # validate feature table only contains unique node_ids
        if feature_table.get_column('node_id').n_unique() != feature_table.height:
            raise ValueError(
                'Candidate feature table contains duplicate node_id values'
            )

        # validate if feature_table contains null values
        if (
            feature_table
            .select(
                pl.any_horizontal(
                    pl.all()
                    .is_null()
                )
                .any()
            )
        ).item():
            raise ValueError(
                'Candidate feature table contains null values'
            )

    ### public candidate feature builder methods
    def build_feature_table(
            self,
            force_recompute: bool = False
    ) -> pl.DataFrame:
        """ Build one label-free feature row for every non-seed candidate """
        # return feature table if it is already in the cache and force recompute is False
        if self._feature_table is not None and not force_recompute:
            return self._feature_table

        # candidate scores
        candidate_scores = (
            self.scores
            .filter(
                ~pl.col('is_seed')
            )
            .drop('is_seed')
            .sort(
                'node_id'
            )
            .collect(
                engine = 'streaming'
            )
        )

        # validate if candidate scores is non-empty
        if candidate_scores.is_empty():
            raise ValueError(
                'The score table contains no non-seed candidates'
            )

        # validate leaked columns present in the candidates scores
        leaked_columns = self.FORBIDDEN_FEATURE_COLUMNS & set(candidate_scores.columns)

        if leaked_columns:
            sorted_leaked_columns = ', '.join(
                sorted(leaked_columns)
            )

            raise ValueError(
                f'Candidate scores contain evaluation labels: {sorted_leaked_columns}'
            )

        # edge features
        edge_features = self._build_edge_feature_table()

        # columns to fill null values with zeros
        zero_fill_columns = [
            'outgoing_transaction_count',
            'outgoing_log_count_weight',
            'incoming_transaction_count',
            'incoming_log_count_weight',
            'forward_seed_neighbor_count',
            'forward_seed_transaction_count',
            'forward_seed_log_count_weight',
            'reverse_seed_neighbor_count',
            'reverse_seed_transaction_count',
            'reverse_seed_log_count_weight'
        ]

        # construct feature table
        feature_table = (
            candidate_scores
            .join(
                edge_features,
                on = 'node_id',
                how = 'left'
            )
            .with_columns(
                [
                    # fill null values
                    pl.col(column_name)
                    .fill_null(0)
                    .alias(
                        column_name
                    )
                    for column_name in zero_fill_columns
                ]
            )
            .with_columns(
                [
                    # is direct forward seed neighbor
                    (
                        pl.col('forward_seed_neighbor_count') > 0
                    )
                    .alias(
                        'is_direct_forward_seed_neighbor'
                    ),
                    # is direct reverse seed neighbor
                    (
                        pl.col('reverse_seed_neighbor_count') > 0
                    )
                    .alias(
                        'is_direct_reverse_seed_neighbor'
                    ),
                    # directional seed neighbor count
                    (
                        pl.col('forward_seed_neighbor_count') + pl.col('reverse_seed_neighbor_count')
                    )
                    .alias(
                        'directional_seed_neighbor_count'
                    ),
                    # directional seed transaction count
                    (
                        pl.col('forward_seed_transaction_count') + pl.col('reverse_seed_transaction_count')
                    )
                    .alias(
                        'directional_seed_transaction_count'
                    ),
                    # total transaction count from edges
                    (
                        pl.col('outgoing_transaction_count') + pl.col('incoming_transaction_count')
                    )
                    .alias(
                        'total_transaction_count_from_edges'
                    ),
                    # directional max personalized PageRank unweighted
                    pl.max_horizontal(
                        self.FORWARD_PPR_COLUMN,
                        self.REVERSE_PPR_COLUMN
                    )
                    .alias(
                        self.MAX_PPR_COLUMN
                    ),
                    # directional min personalized PageRank unweighted
                    pl.min_horizontal(
                        self.FORWARD_PPR_COLUMN,
                        self.REVERSE_PPR_COLUMN
                    )
                    .alias(
                        'directional_min_personalized_pagerank_unweighted'
                    )
                ]
            )
            .with_columns(
                [
                    # degree balance
                    pl.when(
                        pl.col('total_directed_degree') > 0
                    )
                    .then(
                        (
                            pl.col('out_degree').cast(pl.Float64) - pl.col('in_degree').cast(pl.Float64)
                        )
                        / pl.col('total_directed_degree').cast(pl.Float64)
                    )
                    .otherwise(0.0)
                    .alias(
                        'degree_balance'
                    ),
                    # transaction count balance
                    pl.when(
                        pl.col('total_transaction_count_from_edges') > 0
                    )
                    .then(
                        (
                            pl.col('outgoing_transaction_count') - pl.col('incoming_transaction_count')
                        )
                        / pl.col('total_transaction_count_from_edges')
                    )
                    .otherwise(0.0)
                    .alias(
                        'transaction_count_balance'
                    ),
                    # transaction events per neighbor
                    pl.when(
                        pl.col('total_directed_degree') > 0
                    )
                    .then(
                        pl.col('account_transaction_event_count').cast(pl.Float64)
                        / pl.col('total_directed_degree').cast(pl.Float64)
                    )
                    .otherwise(0.0)
                    .alias(
                        'transaction_events_per_neighbor'
                    ),
                    # is bidirectional seed neighbor
                    (
                        pl.col('is_direct_forward_seed_neighbor')
                        & pl.col('is_direct_reverse_seed_neighbor')
                    )
                    .alias(
                        'is_bidirectional_seed_neighbor'
                    )
                ]
            )
        )

        # candidate count
        candidate_count = feature_table.height

        # forward & reverse candidate ranks
        forward_rank = self._rank_candidates(
            feature_table = feature_table,
            score_column = self.FORWARD_PPR_COLUMN,
            rank_column = 'forward_personalized_pagerank_rank'
        )

        reverse_rank = self._rank_candidates(
            feature_table = feature_table,
            score_column = self.REVERSE_PPR_COLUMN,
            rank_column = 'reverse_personalized_pagerank_rank'
        )

        # update feature table by joining forward & reverse ranks
        feature_table = (
            feature_table
            .join(
                forward_rank,
                on = 'node_id',
                how = 'left'
            )
            .join(
                reverse_rank,
                on = 'node_id',
                how = 'left'
            )
            .with_columns(
                [
                    # forward personalized pagerank rank percentile
                    (
                        pl.col('forward_personalized_pagerank_rank').cast(pl.Float64) / float(candidate_count)
                    )
                    .alias(
                        'forward_personalized_pagerank_rank_percentile'
                    ),
                    # reverse personalized pagerank rank percentile
                    (
                        pl.col('reverse_personalized_pagerank_rank').cast(pl.Float64) / float(candidate_count)
                    )
                    .alias(
                        'reverse_personalized_pagerank_rank_percentile'
                    ),
                    # best direction personalized pagerank rank
                    pl.min_horizontal(
                        'forward_personalized_pagerank_rank',
                        'reverse_personalized_pagerank_rank'
                    )
                    .alias(
                        'best_direction_personalized_pagerank_rank'
                    ),
                    # personalized pagerank directional rank gap
                    (
                        pl.col('forward_personalized_pagerank_rank').cast(pl.Int64)
                        - pl.col('reverse_personalized_pagerank_rank').cast(pl.Int64)
                    )
                    .abs()
                    .alias(
                        'personalized_pagerank_directional_rank_gap'
                    ),
                    # personalized pagerank reciprocal rank fusion
                    (
                        1.0
                        / (
                            self.rrf_constant + pl.col('forward_personalized_pagerank_rank')
                        )
                        +
                        1.0
                        / (
                            self.rrf_constant + pl.col('reverse_personalized_pagerank_rank')
                        )
                    )
                    .alias(
                        self.RRF_COLUMN
                    )
                ]
            )
            .sort(
                'node_id'
            )
        )

        # validate feature table
        self._validate_feature_table(
            feature_table = feature_table
        )

        # store feature table in the cache
        self._feature_table = feature_table

        return self._feature_table

    def build_feature_summary(
            self,
            feature_table: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """ Summarize the label-free candidate feature population """
        # feature table
        features = (
            feature_table
            if feature_table is not None
            else self.build_feature_table(
                force_recompute = False
            )
        )

        # validate feature table
        self._validate_feature_table(
            feature_table = features
        )

        return (
            features
            .select(
                [
                    # candidate count
                    pl.len()
                    .alias(
                        'candidate_count'
                    ),
                    # feeature column count
                    pl.lit(len(features.columns))
                    .alias(
                        'feature_column_count'
                    ),
                    # forward reachable candidate count
                    pl.col('is_forward_reachable_from_seed')
                    .sum()
                    .alias(
                        'forward_reachable_candidate_count'
                    ),
                    # reverse reachable candidate count
                    pl.col('is_reverse_reachable_from_seed')
                    .sum()
                    .alias(
                        'reverse_reachable_candidate_count'
                    ),
                    # direct forward seed neighbor count
                    pl.col('is_direct_forward_seed_neighbor')
                    .sum()
                    .alias(
                        'direct_forward_seed_neighbor_count'
                    ),
                    # direct reverse seed neighbor count
                    pl.col('is_direct_reverse_seed_neighbor')
                    .sum()
                    .alias(
                        'direct_reverse_seed_neighbor_count'
                    ),
                    # bidirectional seed neighbor count
                    pl.col('is_bidirectional_seed_neighbor')
                    .sum()
                    .alias(
                        'bidirectional_seed_neighbor_count'
                    )
                ]
            )
        )

    def build_labeled_evaluation_table(
            self,
            feature_table: pl.DataFrame | None = None 
    ) -> pl.DataFrame:
        """ Attach withheld labels after feature construction for evaluation only """
        # feature table
        features = (
            feature_table
            if feature_table is not None
            else self.build_feature_table(
                force_recompute = False
            )
        )

        # validate feature table
        self._validate_feature_table(
            feature_table = features
        )

        # target table
        target_table = (
            self.targets
            .select(
                [
                    'node_id',
                    'ring_id',
                    'target_order'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

        # validate if target table is empty
        if target_table.is_empty():
            raise ValueError(
                'The experiment contains no held-out targets'
            )

        # validate target table contains only unique node ids
        if target_table.get_column('node_id').n_unique() != target_table.height:
            raise ValueError(
                'Held-out target node IDs must be all unique'
            )

        # labeled feature table
        labeled_table = (
            features
            .join(
                target_table,
                on = 'node_id',
                how = 'left'
            )
            .with_columns(
                # is ring target label
                pl.col('ring_id')
                .is_not_null()
                .alias(
                    'is_ring_target'
                )
            )
        )

        # target count
        target_count = (
            labeled_table
            .get_column('is_ring_target')
            .sum()
        )

        # validate all held-out targets present in the candidate feature table
        if int(target_count) != target_table.height:
            raise ValueError(
                'At least one held-out target is absent from the candidate future table'
            )

        return labeled_table

    def evaluate_rankings(
            self,
            score_columns: Sequence[str],
            k_values: Sequence[int] = (100, 500, 1_000, 5_000),
            feature_table: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """ Evaluate label-free scores after ranking all non-seed candidates """
        # validate the instance type of score columns, and remove duplicates if any present
        if isinstance(score_columns, str):
            selected_scores = (score_columns, )
        else:
            selected_scores = tuple(
                dict.fromkeys(score_columns)
            )

        # validate if score columns is non-empty
        if not selected_scores:
            raise ValueError(
                'At least one score column must be requested'
            )

        # deduplicate k_values if present
        selected_k_values = tuple(
            dict.fromkeys(k_values)
        )

        # validate if k values is non-empty
        if not selected_k_values:
            raise ValueError(
                'At least one k value must be requested'
            )

        # validate instance type and non-negativity of k values
        for k_value in selected_k_values:
            if not isinstance(k_value, int) or k_value <= 0:
                raise ValueError(
                    'Every k value must be a positive integer'
                )

        # feature table
        features = (
            feature_table
            if feature_table is not None
            else self.build_feature_table(
                force_recompute = False
            )
        )

        # validate feature table
        self._validate_feature_table(
            feature_table = features
        )

        # unknown score columns
        unknown_scores = set(selected_scores) - set(features.columns)

        # validate if unknown score columns is empty
        if unknown_scores:
            sorted_unknown_scores = ', '.join(
                sorted(unknown_scores)
            )

            raise ValueError(
                f'Unknown score columns: {sorted_unknown_scores}'
            )

        # targets table
        targets = (
            self.targets
            .select(
                [
                    'node_id',
                    'ring_id'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

        # validate if target table is empty
        if targets.is_empty():
            raise ValueError(
                'Ranking evaluation requires at least one held-out target'
            )

        # candidate count
        candidate_count = features.height

        # target count
        target_count = targets.height

        # unique ring count
        ring_count = (
            targets
            .get_column('ring_id')
            .n_unique()
        )

        # construct non-seed candidate rankings
        summary_rows: list[dict[str, str | int | float]] = []

        for score_column in selected_scores:

            # ranked candidates
            ranked_candidates = (
                features
                .select(
                    [
                        'node_id',
                        score_column
                    ]
                )
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
                .with_row_index(
                    name = 'rank',
                    offset = 1
                )
            )

            # target ranks
            target_ranks = (
                targets
                .join(
                    ranked_candidates,
                    on = 'node_id',
                    how = 'left'
                )
            )

            # validate if all target scores are present
            if target_ranks.get_column('rank').null_count() > 0:
                raise ValueError(
                    f'A held-out target is absent from the {score_column!r} ranking'
                )

            # first target rank
            first_target_rank_value = (
                target_ranks
                .get_column('rank')
                .min()
            )

            # validate first target rank value is non-empty
            if first_target_rank_value is None:
                raise ValueError(
                    f'No target ranks are available for {score_column!r}'
                )

            # validate first target rank value instance type
            if not isinstance(first_target_rank_value, int):
                raise TypeError(
                    f'The minimum target rank must be an integer'
                )

            # ensure type casting for first target rank
            first_target_rank = int(first_target_rank_value)

            for k_value in selected_k_values:
                # effective k
                effective_k = min(
                    k_value,
                    candidate_count
                )

                # hits
                hits = (
                    target_ranks
                    .filter(
                        pl.col('rank') <= effective_k
                    )
                )

                # target hits
                target_hits = hits.height

                # ring hits
                ring_hits = (
                    hits
                    .get_column('ring_id')
                    .n_unique()
                )

                summary_rows.append(
                    {
                        'score_name': score_column,
                        'requested_k': k_value,
                        'effective_k': effective_k,
                        'candidate_count': candidate_count,
                        'target_count': target_count,
                        'target_hits_at_k': target_hits,
                        'precision_at_k': (
                            target_hits / effective_k
                        ),
                        'target_recall_at_k': (
                            target_hits / target_count
                        ),
                        'ring_count': ring_count,
                        'rings_hit_at_k': ring_hits,
                        'ring_hit_rate_at_k': (
                            ring_hits / ring_count
                        ),
                        'first_target_rank': first_target_rank
                    }
                )

        return (
            pl.DataFrame(summary_rows)
            .sort(
                [
                    'requested_k',
                    'target_hits_at_k',
                    'rings_hit_at_k',
                    'score_name'
                ],
                descending = [
                    False,
                    True,
                    True,
                    False
                ]
            )
        )

    def validate_feature_file(
            self,
            output_path: Path | str
    ) -> None:
        """ Validate a persisted candidate feature table """
        # get feature path
        feature_path = Path(output_path)

        # validate if feature path exists
        if not feature_path.is_file():
            raise FileNotFoundError(
                f'Candidate feature file does not exist: {feature_path}'
            )

        # validate if feature path is non-empty
        if feature_path.stat().st_size == 0:
            raise ValueError(
                f'Candidate feature file is empty: {feature_path}'
            )

        # validate persisted feature table
        persisted_features = pl.read_parquet(
            source = feature_path
        )

        self._validate_feature_table(
            feature_table = persisted_features
        )

    def write_feature_table(
            self,
            output_path: Path | str,
            overwrite: bool = False
    ) -> Path:
        """ Persist the label-free candidate feature table atomically """
        # validate overwrite instance type
        if not isinstance(overwrite, bool):
            raise TypeError(
                'overwrite must be a boolean instance'
            )

        # get feature table path
        feature_path = Path(output_path)

        # return feature table if it exists and overwrite is False
        if feature_path.exists() and not overwrite:
            return feature_path

        # create feature path parent directory
        feature_path.parent.mkdir(
            parents = True,
            exist_ok = True
        )

        # temporary path,
        temporary_path = feature_path.with_name(
            name = f".{feature_path.stem}.tmp{feature_path.suffix}"
        )

        # dump temporary path
        temporary_path.unlink(
            missing_ok = True
        )

        try:
            # write parquet file to temporary path
            self.build_feature_table().write_parquet(
                file = temporary_path,
                compression = 'zstd',
                statistics = True
            )

            # validate temporary file
            self.validate_feature_file(
                output_path = temporary_path
            )

            # replace temporary file with final file
            temporary_path.replace(
                target = feature_path
            )

            # validate the final feature table
            self.validate_feature_file(
                output_path = feature_path
            )

        except Exception:
            # dump temporary path
            temporary_path.unlink(
                missing_ok = True
            )

            raise

        return feature_path