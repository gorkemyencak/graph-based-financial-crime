import polars as pl

SAML_D_REQUIRED_COLUMNS: tuple[str, ...] = (
    'Time',
    'Date',
    'Sender_account',
    'Receiver_account',
    'Amount',
    'Payment_currency',
    'Received_currency',
    'Sender_bank_location',
    'Receiver_bank_location',
    'Payment_type',
    'Is_laundering',
    'Laundering_type'
)

SAML_D_SCHEMA_OVERRIDES = {
    'Time': pl.String,
    'Date': pl.String,
    'Sender_account': pl.String,
    'Receiver_account': pl.String,
    'Amount': pl.Float64,
    'Payment_currency': pl.String,
    'Received_currency': pl.String,
    'Sender_bank_location': pl.String,
    'Receiver_bank_location': pl.String,
    'Payment_type': pl.String,
    'Is_laundering': pl.Int8,
    'Laundering_type': pl.String
}

SAML_D_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'