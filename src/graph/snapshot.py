import re
import polars as pl

from datetime import datetime
from pathlib import Path

from src.graph.temporal_split import SAMLDTemporalSplitter
from src.graph.builder import SAMLDGraphTableBuilder

class GraphSnapshotPaths:
    """ Paths belonging to one persisted graph snapshot """
    def __init__(
            self,
            directory: Path | str,
            nodes: Path | str,
            edges: Path | str,
            seeds: Path | str,
            targets: Path | str
    ) -> None:
        # attributes
        self.directory = Path(directory)
        self.nodes = Path(nodes)
        self.edges = Path(edges)
        self.seeds = Path(seeds)
        self.targets = Path(targets)

    def as_dict(self) -> dict[str, Path]:
        """ Return table names and their Parquet paths """
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'seeds': self.seeds,
            'targets': self.targets
        }


class SAMLDGraphSnapshotWriter:
    """  
    Persist leakage-safe graph snapshots and their evaluation tables

    The graph tables contain only historical, label-free information. Known suspicious accounts and future targets are written
    to separate files so later PageRank code cannot accidentally treat labels as graph features    
    """
    # define valid reg-expressions when creating valid snapshots
    _VALID_SNAPSHOT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

    # specifying minimum columns each persisted table must contain
    _REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
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
        'seeds': (
            'node_id',
            'account',
            'first_known_suspicious_timestamp',
            'last_known_suspicious_timestamp'
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

    def __init__(
            self,
            graph_builder: SAMLDGraphTableBuilder,
            temporal_splitter: SAMLDTemporalSplitter,
            output_dir: Path
    ) -> None:
        # attributes
        self.graph_builder = graph_builder
        self.temporal_splitter = temporal_splitter
        self.output_dir = Path(output_dir)

    ### private helper methods
    @classmethod
    def _validate_snapshot_name(
        cls,
        snapshot_name: str
    ) -> None:
        """ Validate that a snapshot is safe to use as a directory name """
        # type validation
        if not isinstance(snapshot_name, str):
            raise TypeError(
                'snapshot_name must be a string.'
            )

        # pattern validation
        if cls._VALID_SNAPSHOT_NAME.fullmatch(snapshot_name) is None:
            raise ValueError(
                'Invalid snapshot name: check the allowed regular epressions'
            )

    @staticmethod
    def _validate_window(
        cutoff: datetime,
        target_end_exclusive: datetime
    ) -> None:
        """ Validate the graph cutoff and future target boundary """
        # cutoff type validation
        if not isinstance(cutoff, datetime):
            raise TypeError(
                'cutoff must be a datetime instance'
            )

        # target_end_exclusive type validation
        if not isinstance(target_end_exclusive, datetime):
            raise TypeError(
                'target_end_exclusive must be a datetime instance'
            )

        # ensure the logical expreassion 
        if cutoff >= target_end_exclusive:
            raise ValueError(
                'cutoff must be earlier than target_end_exclusive'
            )

    @staticmethod
    def _get_temporary_path(
        path: Path
    ) -> Path:
        """ Return a temporary sibling path for one Parquet file """
        return path.with_name(
            f'.{path.stem}.tmp{path.suffix}'
        )

    @staticmethod
    def _sink_parquet(
        lazy_frame: pl.LazyFrame,
        path: Path
    ) -> None:
        """ Stream one lazy query to a compressed Parquet file """
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
        """ Perform file-level and schema-level checks on one saved table """
        # check if the path exists
        if not path.is_file():
            raise FileNotFoundError(
                f'Snapshot table does not exist: {path}'
            )

        # check the file size of path
        if path.stat().st_size == 0:
            raise ValueError(
                f'Snapshot table is empty: {path}'
            )

        # get actual columns on parquet table
        actual_columns = set(
            pl.scan_parquet(source = path)
            .collect_schema()
            .names()
        )

        # expected columns in the schema
        required_columns = set(
            cls._REQUIRED_COLUMNS[table_name]
        )

        # missing columns not exist on the actual schema
        missing_columns = required_columns - actual_columns

        # check missing columns, and return sorted missing columns if present
        if missing_columns:
            sorted_missing_columns = ', '.join(
                sorted(missing_columns)
            )

            raise ValueError(
                f'{table_name!r} snapshot schema has missing columns: '
                f'{sorted_missing_columns}'
            )

    @classmethod
    def _validate_snapshot_files(
        cls,
        paths: dict[str, Path]
    ) -> None:
        """ Validate the four persisted tables for one snapshot """
        for table_name, path in paths.items():
            cls._validate_parquet_file(
                table_name = table_name,
                path = path
            )

    def _build_seed_table(
            self,
            cutoff: datetime,
            node_index: pl.LazyFrame
    ) -> pl.LazyFrame:
        """ Map historical PageRank seed accounts to snapshot node IDs """
        return (
            self.temporal_splitter
            .build_known_suspicious_accounts(
                cutoff = cutoff
            )
            .join(
                node_index,
                on = 'account',
                how = 'left'
            )
            .select(
                [
                    'node_id',
                    'account',
                    'first_known_suspicious_timestamp',
                    'last_known_suspicious_timestamp',
                    'laundering_sender_event_count',
                    'laundering_receiver_event_count',
                    'laundering_account_event_count'
                ]
            )
            .sort('node_id')
        )

    def _build_target_table(
            self,
            cutoff: datetime,
            target_end_exclusive: datetime,
            node_index: pl.LazyFrame
    ) -> pl.LazyFrame:
        """ Map all future suspicious accounts to snapshot node IDs when seen """
        return (
            self.temporal_splitter
            .build_future_suspicious_accounts(
                window_start = cutoff,
                window_end_exclusive = target_end_exclusive
            )
            .join(
                node_index,
                on = 'account',
                how = 'left'
            )
            .select(
                [
                    'node_id',
                    'account',
                    'first_future_suspicious_timestamp',
                    'last_future_suspicious_timestamp',
                    'laundering_sender_event_count',
                    'laundering_receiver_event_count',
                    'laundering_account_event_count',
                    'known_before_window',
                    'seen_before_window',
                    'is_new_suspicious_account',
                    'is_rankable_new_suspicious_account'
                ]
            )
            .sort(
                [
                    'first_future_suspicious_timestamp',
                    'account'
                ]
            )
        )


    ### public snapshot writer methods
    def get_snapshot_paths(
            self,
            snapshot_name: str
    ) -> GraphSnapshotPaths:
        """ Return the standard paths for one snapshot """
        # validate snapshot name
        self._validate_snapshot_name(
            snapshot_name = snapshot_name
        )

        # define snapshot dir
        snapshot_dir = self.output_dir / snapshot_name

        return GraphSnapshotPaths(
            directory = snapshot_dir,
            nodes = snapshot_dir / 'nodes.parquet',
            edges = snapshot_dir / 'edges.parquet',
            seeds = snapshot_dir / 'seeds.parquet',
            targets = snapshot_dir / 'targets.parquet'
        )

    def validate_snapshot(
            self,
            snapshot_name: str
    ) -> None:
        """ Validate existing snapshot files without rebuilding them """
        # standard paths for one snapshot
        paths = self.get_snapshot_paths(
            snapshot_name = snapshot_name
        )

        # validate snapshot files
        self._validate_snapshot_files(
            paths = paths.as_dict()
        )

    def scan_snapshot(
            self,
            snapshot_name: str
    ) -> dict[str, pl.LazyFrame]:
        """ Lazily scan all four tables from an existing snapshot """
        # validate existing snapshot files
        self.validate_snapshot(
            snapshot_name = snapshot_name
        )

        return {
            table_name: pl.scan_parquet(source = path)
            for table_name, path in (
                self.get_snapshot_paths(
                    snapshot_name = snapshot_name
                )
                .as_dict()
                .items()
            )
        }

    def write_snapshot(
            self,
            snapshot_name: str,
            cutoff: datetime,
            target_end_exclusive: datetime,
            overwrite: bool = False
    ) -> GraphSnapshotPaths:
        """
        Build and persist one graph snapshot

        All four temporary tables must be written and validated before their final paths are replaced      
        """
        # validate snapshot name
        self._validate_snapshot_name(
            snapshot_name = snapshot_name
        )

        # validate the graph cutoff and future target boundary
        self._validate_window(
            cutoff = cutoff,
            target_end_exclusive = target_end_exclusive
        )

        # standard paths for one snapshot
        paths = self.get_snapshot_paths(
            snapshot_name = snapshot_name
        )

        # standard paths in dictionary format
        final_paths = paths.as_dict()

        # existing tables
        existing_tables = [
            table_name
            for table_name, path in final_paths.items()
            if path.exists()
        ]

        # check for incomplete snapshot tables
        if existing_tables and not overwrite:
            if len(existing_tables) == len(final_paths):
                # validate snapshot files
                self._validate_snapshot_files(
                    paths = final_paths
                )

                return paths

            existing_tables_text = ', '.join(existing_tables)
            raise FileExistsError(
                f'Snapshot {snapshot_name!r} is incomplete. Existing tables: {existing_tables_text}. '
                'Re-run with overwrite=True to rebuild it.'
            )

        # create paths directory
        paths.directory.mkdir(
            parents = True,
            exist_ok = True 
        )

        # unlink temporary files if present
        temporary_paths = {
            table_name: self._get_temporary_path(path = path)
            for table_name, path in final_paths.items()
        }

        for temporary_path in temporary_paths.values():
            temporary_path.unlink(
                missing_ok = True
            )

        try:
            # account-node table
            nodes = (
                self.graph_builder
                .build_node_table(
                    cutoff = cutoff
                )
            )

            # aggregated edge per directed account pair
            edges = (
                self.graph_builder
                .build_edge_table(
                    cutoff = cutoff
                )
            )

            # dump node and edge tables to the temporary path
            self._sink_parquet(
                lazy_frame = nodes,
                path = temporary_paths['nodes']
            )

            self._sink_parquet(
                lazy_frame = edges,
                path = temporary_paths['edges']
            )

            # node-index table
            node_index = (
                pl.scan_parquet(
                    temporary_paths['nodes']
                )
                .select(
                    [
                        'account',
                        'node_id'
                    ]
                )
            )

            # seeds query plan
            seeds = self._build_seed_table(
                cutoff = cutoff,
                node_index = node_index
            )

            # targets query plan
            targets = self._build_target_table(
                cutoff = cutoff,
                target_end_exclusive = target_end_exclusive,
                node_index = node_index
            )

            # dump seeds and targets to the temporary path
            self._sink_parquet(
                lazy_frame = seeds,
                path = temporary_paths['seeds']
            )

            self._sink_parquet(
                lazy_frame = targets,
                path = temporary_paths['targets']
            )

            # validate temporary files
            self._validate_snapshot_files(
                paths = temporary_paths
            )

            # replace temporary files with final 
            for table_name, final_path in final_paths.items():
                temporary_paths[table_name].replace(final_path)

        except Exception:
            # unlink temporary files 
            for temporary_path in temporary_paths.values():
                temporary_path.unlink(
                    missing_ok = True
                )

            raise

        # validate final tables
        self._validate_snapshot_files(
            paths = final_paths
        )

        return paths

    def build_snapshot_file_summary(
            self,
            snapshot_name: str,
            snapshot: dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """ Summary table of presenting snapshot table sizes for persisted snapshots """
        return (
            pl.concat(
                [
                    lazy_frame
                    .select(
                        [
                            # snapshot
                            pl.lit(snapshot_name)
                            .alias(
                                'snapshot'
                            ),
                            # table
                            pl.lit(table_name)
                            .alias(
                                'table'
                            ),
                            # table size
                            pl.len()
                            .cast(pl.UInt64)
                            .alias(
                                'row_count'
                            )
                        ]
                    )
                    for table_name, lazy_frame in snapshot.items()
                ],
                how = 'vertical'
            )
        )

    def build_saved_target_summary(
            self,
            snapshot_name: str,
            targets: pl.LazyFrame
    ) -> pl.LazyFrame:
        """ Summarize the persisted evaluation targets """
        return (
            targets
            .select(
                [
                    # snapshot
                    pl.lit(snapshot_name)
                    .alias(
                        'snapshot'
                    ),
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
                    # rankable new suspicious account count
                    pl.col('is_rankable_new_suspicious_account')
                    .sum()
                    .cast(pl.UInt64)
                    .alias(
                        'rankable_new_suspicious_account_count'
                    ),
                    # target missing node id count
                    pl.col('node_id')
                    .null_count()
                    .cast(pl.UInt64)
                    .alias(
                        'target_without_node_id_count'
                    )
                ]
            )
        )