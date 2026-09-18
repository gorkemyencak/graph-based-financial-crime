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
            'account_transaction_count'
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
        'ring_account': {
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
                    int(forward_min)
                    if np.isfinite(forward_min)
                    else None,
                    int(reverse_min)
                    if np.isfinite(reverse_min)
                    else None,
                )
            )

        return result      

