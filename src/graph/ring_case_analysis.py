import numpy as np
import polars as pl
import igraph as ig

from pathlib import Path
from typing import Sequence

class SAMLDRingCaseAnalyzer:
    """
    Compare pooler PPR rankings and trace paths from each ring's own seeds

    The saved node scores were calculated in notebook 7. Target and ring labels are joined only after ranking. Path
    checks are retrospective diagnostics, not inputs to the saved PPR scores. A path from an own-ring seed does not
    identify the exact source of pooled PPR probability
    """
    SCORE_COLUMNS: dict[str, str] = {
        'forward': 'personalized_pagerank_unweighted',
        'reverse': 'reverse_personalized_pagerank_unweighted',
        'bidirectional': 'bidirectional_personalized_pagerank_unweighted'
    }

    REQUIRED_COLUMNS: dict[str, set[str]] = {
        'nodes': {
            'node_id',
            'out_degree',
            'in_degree',
            'account_transaction_event_count'
        },
        'edges': {
            'source_node_id',
            'target_node_id',
            'count_weight'
        },
        'rings': {
            'ring_id',
            'ring_size',
            'seed_count',
            'target_count'
        },
        'ring_accounts': {
            'ring_id',
            'node_id',
            'account',
            'ring_member_order',
            'is_ring_seed',
            'is_ring_target'
        },
        'seeds': {
            'ring_id',
            'node_id'
        },
        'targets': {
            'ring_id',
            'node_id',
            'account',
            'target_order'
        },
        'scores': {
            'node_id',
            'is_seed',
            *SCORE_COLUMNS.values()
        }
    }

    def __init__(
            self,
            experiment_dir: Path | str,
            k: int = 1_000,
            score_path: Path | str | None = None
    ) -> None:
        # validate whether k is a non-negative integer
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                'k must be a positive integer'
            )
        
        # attributes
        self.experiment_dir = Path(experiment_dir)
        self.score_path = (
            Path(score_path)
            if score_path is not None
            else self.experiment_dir / 'personalized_pagerank_scores.parquet'
        )
        self.k = k

        # define parquet and score files
        self.files = {
            'nodes': self.experiment_dir / 'nodes.parquet',
            'edges': self.experiment_dir / 'edges.parquet',
            'rings': self.experiment_dir / 'rings.parquet',
            'ring_accounts': self.experiment_dir / 'ring_accounts.parquet',
            'seeds': self.experiment_dir / 'seeds.parquet',
            'targets': self.experiment_dir / 'targets.parquet',
            'scores': self.score_path
        }

        # validate parquet and score files
        self._validate_parquet_files(
            files = self.files
        )

        # validate required columns
        self._validate_required_columns(
            columns = self.REQUIRED_COLUMNS
        )

        # file attributes
        score_columns = [
            'node_id',
            'is_seed',
            *self.SCORE_COLUMNS.values()
        ]
        self.scores = (
            pl.scan_parquet(source = self.score_path)
            .select(
                score_columns
            )
        )
        self.nodes = pl.scan_parquet(source = self.files['nodes'])
        self.edges = pl.scan_parquet(source = self.files['edges'])
        self.rings = pl.read_parquet(source = self.files['rings'])
        self.ring_accounts = pl.read_parquet(source = self.files['ring_accounts'])
        self.seeds = pl.read_parquet(source = self.files['seeds'])
        self.targets = pl.read_parquet(source = self.files['targets'])

        # private class attributes
        self._candidate_scores: pl.DataFrame | None = None
        self._target_ranks: pl.DataFrame | None = None
        self._ring_comparison: pl.DataFrame | None = None
        self._graph: ig.Graph | None = None

    ### private validation methods
    @classmethod
    def _validate_parquet_files(
        cls,
        files: dict[str, Path]
    ) -> None:
        """ Validate persisted Parquet files """
        # validate if parquet files exist
        for name, path in files.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f'Missing {name} file: {path}. Run notebooks 06 and 07 first!'
                )

    def _validate_required_columns(
            self,
            columns: dict[str, set[str]]
    ) -> None:
        """ Validate required columns in the persisted Parquet files """
        # validate missing columns if present 
        for name, path in self.files.items():
            # actual columns extracted from persisted files
            actual_columns = set(
                pl.scan_parquet(
                    source = path
                )
                .collect_schema()
                .names()
            )

            # missing columns
            missing_columns = columns[name] - actual_columns

            if missing_columns:
                sorted_missing_columns = ', '.join(
                    sorted(missing_columns)
                )

                raise ValueError(
                    f'{name!r} is missing Personalized PageRank columns: {sorted_missing_columns}'
                )

    ### private score and graph methods
    def _get_candidate_scores(self) -> pl.DataFrame:
        """ Read only the three scores needed for case comparison """
        # compute candidate scores
        if self._candidate_scores is None:
            # candidates
            candidates = (
                self.scores
                .filter(
                    ~pl.col('is_seed')
                )
                .drop('is_seed')
                .collect(
                    engine = 'streaming'
                )
            )

            # validate if candidates table is empty
            if candidates.is_empty():
                raise ValueError(
                    'No non-seed candidates in the score file'
                )

            self._candidate_scores = candidates

        return self._candidate_scores

    def _members(
            self,
            ring_id: int
    ) -> pl.DataFrame:
        """ Return ring members for a matching ring_id """
        # members
        members = (
            self.ring_accounts
            .filter(
                pl.col('ring_id') == ring_id
            )
        )

        # validate if members are empty
        if members.is_empty():
            raise ValueError(
                f'Unknown ring_id: {ring_id}'
            )

        return members

    @staticmethod
    def _distance_pairs(
        graph: ig.Graph,
        sources: list[int],
        targets: list[int]
    ) -> list[tuple[int | None, int | None]]:
        """ Minimum directed hops source->target and target->source per target """
        # forward distances
        forward = np.asarray(
            graph.distances(
                source = sources,
                target = targets,
                mode = 'out'
            ),
            dtype = np.float64
        )

        # reverse distances
        reverse = np.asarray(
            graph.distances(
                source = targets,
                target = sources,
                mode = 'out'
            )
        )

        # directed min distances per target
        result: list[tuple[int | None, int | None]] = []

        for target_index in range(len(targets)):
            # forward min distance
            forward_min = float(
                np.min(
                    forward[:, target_index]
                )
            )

            # reverse min distance
            reverse_min = float(
                np.min(
                    reverse[target_index, :]
                )
            )

            result.append(
                (
                    # forward hop
                    int(forward_min)
                    if np.isfinite(forward_min)
                    else None,
                    # reverse hop
                    int(reverse_min)
                    if np.isfinite(reverse_min)
                    else None,
                )
            )

        return result

    ### public evaluation methods
    def build_target_rank_table(self) -> pl.DataFrame:
        """ Sort non-seed accounts first, then attach withheld target labels """
        # return target ranks if it is already stored in the cache
        if self._target_ranks is not None:
            return self._target_ranks

        # candidate scores based on non-seed accounts
        candidates = self._get_candidate_scores()

        # ranked targets
        ranked_targets = (
            self.targets
            .select(
                [
                    'ring_id',
                    'node_id',
                    'account',
                    'target_order'
                ]
            )
        )

        for direction, score_name in self.SCORE_COLUMNS.items():
            # ranked candidates
            ranked_candidates = (
                candidates
                .select(
                    [
                        'node_id',
                        score_name
                    ]
                )
                .sort(
                    [
                        score_name,
                        'node_id'
                    ],
                    descending = [
                        True,
                        False
                    ]
                )
                .with_row_index(
                    name = f'{direction}_rank', 
                    offset = 1
                )
                .rename(
                    {
                        score_name: f'{direction}_score'
                    }
                )
            )

            # update ranked targets with direction scores
            ranked_targets = (
                ranked_targets
                .join(
                    ranked_candidates,
                    on = 'node_id',
                    how = 'left'
                )
            )

        # ensure that none of targets are absent from candidate rankings
        if ranked_targets.select(
            pl.any_horizontal(
                pl.all()
                .is_null()
            )
            .any()
        ).item():
            raise ValueError(
                'A withheld target is absent from the candidate rankings'
            )

        # store in the cache
        self._target_ranks = (
            ranked_targets
            .sort(
                [
                    'ring_id',
                    'target_order'
                ]
            )
        )

        return self._target_ranks

    def build_ring_comparison(self) -> pl.DataFrame:
        """ Count each ring's targets in the global top-k of three rankings """
        # return ring comparison if it is already stored in the cache
        if self._ring_comparison is not None:
            return self._ring_comparison

        # target rank table
        target_ranks = self.build_target_rank_table()

        # target rank table grouped by ring ids
        grouped_target_ranks = (
            target_ranks
            .group_by(
                'ring_id'
            )
            .agg(
                [
                    # direction hits
                    (
                        pl.col(f'{direction}_rank') <= self.k
                    )
                    .sum()
                    .alias(
                        f'{direction}_hits'
                    )
                    for direction in self.SCORE_COLUMNS
                ]
                +
                [
                    # direction first rank
                    pl.col(f'{direction}_rank')
                    .min()
                    .alias(
                        f'{direction}_first_rank'
                    )
                    for direction in self.SCORE_COLUMNS
                ]
            )
        )

        # ring comparison table
        ring_comparison = (
            self.rings
            .select(
                [
                    'ring_id',
                    'ring_size',
                    'seed_count',
                    'target_count'
                ]
            )
            .join(
                grouped_target_ranks,
                on = 'ring_id',
                how = 'left'
            )
            .with_columns(
                # blend gain
                (
                    pl.col('bidirectional_hits').cast(pl.Int64) - pl.col('forward_hits').cast(pl.Int64)
                )
                .alias(
                    'blend_gain'
                ),
                # forward ring recall
                (
                    pl.col('forward_hits') / pl.col('target_count')
                )
                .alias(
                    'forward_ring_recall'
                ),
                # bidirectional ring recall
                (
                    pl.col('bidirectional_hits') / pl.col('target_count')
                )
                .alias(
                    'bidirectional_ring_recall'
                )
            )
            .with_columns(
                # forward status: complete, partial or missed
                pl.when(
                    pl.col('forward_hits') == pl.col('target_count')
                )
                .then(
                    pl.lit('complete')
                )
                .when(
                    pl.col('forward_hits') > 0
                )
                .then(
                    pl.lit('partial')
                )
                .otherwise(
                    pl.lit('missed')
                )
                .alias(
                    'forward_status'
                )
            )
            .sort(
                'ring_id'
            )
        )

        # store in the cache
        self._ring_comparison = ring_comparison

        return self._ring_comparison

    def select_representative_cases(self) -> pl.DataFrame:
        """ Select distinct success, blend-gain, reverse-rescue, and missed rings """
        # ring comparison summary
        summary = self.build_ring_comparison()

        # scenarios: case type, condition, sort columns, descending
        scenarios = [
            (
                'forward_complete',
                pl.col('forward_hits') == pl.col('target_count'),
                ['target_count', 'forward_first_rank', 'ring_id'],
                [True, False, False]
            ),
            (
                'blend_gain',
                pl.col('blend_gain') > 0,
                ['blend_gain', 'bidirectional_hits', 'ring_id'],
                [True, True, False]
            ),
            (
                'reverse_rescue',
                (pl.col('forward_hits') == 0) & (pl.col('reverse_hits') > 0),
                ['reverse_hits', 'ring_id'],
                [True, False]
            ),
            (
                'missed_by_all',
                (pl.col('forward_hits') == 0) & (pl.col('reverse_hits') == 0) & (pl.col('bidirectional_hits') == 0),
                ['target_count', 'ring_id'],
                [True, False]
            )
        ]

        # selected case scenarios
        chosen: list[dict[str, object]] = []
        used: set[int] = set()

        for case_type, condition, sort_columns, descending in scenarios:

            # filtered & sorted ring comparison summary
            filtered_summary = (
                summary
                .filter(
                    condition
                    & ~pl.col('ring_id').is_in(sorted(used))
                )
                .sort(
                    by = sort_columns,
                    descending = descending
                )
            )

            # validate if filtered ring comparison summary is empty
            if filtered_summary.is_empty():
                continue

            filtered_summary_row = filtered_summary.row(
                index = 0,
                named = True
            )

            # add ring_id to 'used' set
            used.add(
                int(filtered_summary_row['ring_id'])
            )

            # append case type and row values to 'chosen' list
            chosen.append(
                {
                    'case_type': case_type,
                    **filtered_summary_row
                }
            )

        # validate whether 'chosen' list is empty
        if not chosen:
            raise ValueError(
                'No rings available for the selected case scenarios'
            )

        return pl.DataFrame(chosen)

    def build_igraph(self) -> ig.Graph:
        """ Load the observation graph once, only for case path tracing """
        # validate if graph is not stored in the cache yet
        if self._graph is None:
            # node count
            node_count = int(
                self.nodes
                .select(
                    pl.len()
                )
                .collect()
                .item()
            )

            # edge table
            edge_table = (
                self.edges
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

            # endpoints
            endpoints = edge_table.iter_rows()

            # store graph object in the cache
            self._graph = ig.Graph(
                n = node_count,
                edges = list(endpoints),
                directed = True
            )

        return self._graph

    def get_ring_edges(
            self,
            ring_id: int
    ) -> pl.DataFrame:
        """ Get observation edges whose two endpoints are ring members """
        # member ids of a particular ring id
        member_ids = (
            self._members(
                ring_id = ring_id
            )
            .get_column(
                'node_id'
            )
            .to_list()
        )

        return (
            self.edges
            .filter(
                pl.col('source_node_id').is_in(member_ids)
                & pl.col('target_node_id').is_in(member_ids)
            )
            .select(
                [
                    'source_node_id',
                    'target_node_id',
                    'count_weight'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

    def build_same_ring_path_table(
            self,
            ring_ids: Sequence[int]
    ) -> pl.DataFrame:
        """  
        Trace selected target paths from the seeds of their own ring

        Whole-graph paths may use intermediary nodes outside the ring. Paths computed on the induced graph stay 
        strictly within its member accounts. No ring-specific PageRank computation is performed        
        """
        # unique ring ids
        unique_ring_ids = list(
            dict.fromkeys(
                int(ring_id)
                for ring_id in ring_ids
            )
        )

        # validate if unique ring ids is empty
        if not unique_ring_ids:
            raise ValueError(
                'Choose at least one ring_id'
            )

        # build igraph
        graph = self.build_igraph()

        # target rank table
        rank_table = self.build_target_rank_table()

        # construct same ring path table
        rows: list[dict[str, int | bool | None]] = []

        for ring_id in unique_ring_ids:
            # ring members
            members = self._members(
                ring_id = ring_id
            )

            # ring member ids
            member_ids = [
                int(node_id)
                for node_id in members['node_id'].to_list()
            ]
            
            # seed ids
            seed_ids = [
                int(node_id)
                for node_id in (
                    self.seeds
                    .filter(
                        pl.col('ring_id') == ring_id
                    )
                    .get_column(
                        'node_id'
                    )
                    .to_list()
                )
            ]

            # ring targets
            ring_targets = (
                self.targets
                .filter(
                    pl.col('ring_id') == ring_id
                )
            )

            # target ids
            target_ids = [
                int(node_id)
                for node_id in ring_targets['node_id'].to_list()
            ]

            # validate if seed_ids and target_ids is empty
            if not seed_ids or not target_ids:
                raise ValueError(
                    f'Ring {ring_id} has no seeds or no targets'
                )

            # global paths
            global_paths = self._distance_pairs(
                graph = graph,
                sources = seed_ids,
                targets = target_ids
            )

            # local index
            local_index = {
                node_id: index
                for index, node_id in enumerate(member_ids)
            }

            # ring member edges
            member_edges = self.get_ring_edges(
                ring_id = ring_id
            )

            # local graph
            local_graph = ig.Graph(
                n = len(member_ids),
                edges = [
                    (local_index[int(source)], local_index[int(target)])
                    for source, target, _ in member_edges.iter_rows()
                ],
                directed = True
            )

            # local paths
            local_paths = self._distance_pairs(
                graph = local_graph,
                sources = [
                    local_index[node_id]
                    for node_id in seed_ids
                ],
                targets = [
                    local_index[node_id]
                    for node_id in target_ids
                ]
            )

            # generate path table row for each ring_id
            for node_id, global_pair, local_pair in zip(
                target_ids,
                global_paths,
                local_paths,
                strict = True
            ):
                rows.append(
                    {
                        'ring_id': ring_id,
                        'node_id': node_id,
                        'full_forward_hops': global_pair[0],
                        'full_reverse_hops': global_pair[1],
                        'within_ring_forward_hops': local_pair[0],
                        'within_ring_reverse_hops': local_pair[1]
                    }
                )

        # path table
        path_table = pl.DataFrame(rows)

        return (
            rank_table
            .join(
                path_table,
                on = [
                    'ring_id',
                    'node_id'
                ],
                how = 'inner'
            )
            .with_columns(
                [
                    # forward hit
                    (
                        pl.col('forward_rank') <= self.k
                    )
                    .alias(
                        'forward_hit'
                    ),
                    # reverse hit
                    (
                        pl.col('reverse_rank') <= self.k
                    )
                    .alias(
                        'reverse_hit'
                    ),
                    # bidirectional hit
                    (
                        pl.col('bidirectional_rank') <= self.k
                    )
                    .alias(
                        'bidirectional_hit'
                    ),
                    # own seed forward path
                    pl.col('full_forward_hops')
                    .is_not_null()
                    .alias(
                        'own_seed_forward_path'
                    ),
                    # own seed reverse path
                    pl.col('full_reverse_hops')
                    .is_not_null()
                    .alias(
                        'own_seed_reverse_path'
                    ),
                    # own seed forward path within ring
                    pl.col('within_ring_forward_hops')
                    .is_not_null()
                    .alias(
                        'own_seed_forward_path_within_ring'
                    ),
                    # own seed reverse path within ring
                    pl.col('within_ring_reverse_hops')
                    .is_not_null()
                    .alias(
                        'own_seed_reverse_path_within_ring'
                    )
                ]
            )
            .sort(
                [
                    'ring_id',
                    'target_order'
                ]
            )
        )
    
    def get_ring_nodes(
            self,
            ring_id: int
    ) -> pl.DataFrame:
        """ Return roles, activity, scores, and global ranks of ring members """
        # ring members
        members = self._members(
            ring_id = ring_id
        )

        # ring member ids
        member_ids = members['node_id'].to_list()

        # ring member activity
        activity = (
            self.nodes
            .filter(
                pl.col('node_id').is_in(member_ids)
            )
            .select(
                [
                    'node_id',
                    'in_degree',
                    'out_degree',
                    'account_transaction_event_count'
                ]
            )
            .collect(
                engine = 'streaming'
            )
        )

        # ring member scores
        scores = (
            self.scores
            .filter(
                pl.col('node_id').is_in(member_ids)
            )
            .drop('is_seed')
            .collect(
                engine = 'streaming'
            )
        )

        # target rank table
        ranks = (
            self.build_target_rank_table()
            .select(
                [
                    'node_id',
                    'forward_rank',
                    'reverse_rank',
                    'bidirectional_rank'
                ]
            )
        )

        return (
            members
            .select(
                [
                    'ring_id',
                    'node_id',
                    'account',
                    'ring_member_order',
                    'is_ring_seed',
                    'is_ring_target'
                ]
            )
            .join(
                activity,
                on = 'node_id',
                how = 'left'
            )
            .join(
                scores,
                on = 'node_id',
                how = 'left'
            )
            .join(
                ranks,
                on = 'node_id',
                how = 'left'
            )
            .sort(
                'ring_member_order'
            )
        )

    def get_top_non_targets(
            self,
            top_n: int = 10
    ) -> pl.DataFrame:
        """ Inspect high-ranked accounts outside the held-out target set """
        # validate whether top_n is an integer instance or non-negative
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(
                'top_n must be a positive integer'
            )

        # forward score column
        forward_score = self.SCORE_COLUMNS['forward']

        # ranked candidate scores of forward score column
        ranked_candidates = (
            self._get_candidate_scores()
            .sort(
                [
                    forward_score,
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

        # ranked candidates not listed in targets
        ranked_candidates_non_targets = (
            ranked_candidates
            .join(
                self.targets
                .select(
                    'node_id'
                ),
                on = 'node_id',
                how = 'anti'
            )
            .sort(
                'global_rank'
            )
            .head(
                top_n
            )
        )

        return (
            ranked_candidates_non_targets
            .join(
                self.nodes
                .select(
                    [
                        'node_id',
                        'account',
                        'in_degree',
                        'out_degree',
                        'account_transaction_event_count'
                    ]
                )
                .filter(
                    pl.col('node_id').is_in(
                        ranked_candidates_non_targets['node_id']
                        .to_list()
                    )
                )
                .collect(
                    engine = 'streaming'
                ),
                on = 'node_id',
                how = 'left'
            )
            .select(
                [
                    'global_rank',
                    'node_id',
                    'account',
                    forward_score,
                    'in_degree',
                    'out_degree',
                    'account_transaction_event_count'
                ]
            )
        )