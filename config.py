from pathlib import Path

from datetime import datetime

def get_project_root(
        start_path: Path
) -> Path:
    """
    Return the project root by searching upward for pyproject.toml
    
    To make the path robust when executing the code from scripts and notebooks
    """
    current_path = start_path.resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / 'pyproject.toml').exists():
            return parent

    raise FileNotFoundError('Project root could not be found. Please make sure that pyproject.toml exists in the project folder!')

PROJECT_ROOT = get_project_root(Path(__file__))

DATA_DIR = PROJECT_ROOT / 'data'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
METRICS_DIR = REPORTS_DIR / 'metrics'


SAML_D_DATASET_HANDLE = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
SAML_D_FILE_NAME = "SAML-D.csv"

SAML_D_RAW_PATH = RAW_DATA_DIR / SAML_D_FILE_NAME
SAML_D_INTERIM_PATH = INTERIM_DATA_DIR / 'saml_d_transactions.parquet'

# historical, validation, and test windows
SAML_D_VALIDATION_START = datetime(
    year = 2023,
    month = 6, 
    day = 1
)

SAML_D_TEST_START = datetime(
    year = 2023,
    month = 7, 
    day = 1
)

SAML_D_TEST_END_EXCLUSIVE = datetime(
    year = 2023,
    month = 8,
    day = 23
)