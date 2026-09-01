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

SAML_D_COLUMN_RENAME_MAP: dict[str, str] = {
    'Time': 'time',
    'Date': 'date',
    'Sender_account': 'sender_account',
    'Receiver_account': 'receiver_account',
    'Amount': 'amount',
    'Payment_currency': 'payment_currency',
    'Received_currency': 'received_currency',
    'Sender_bank_location': 'sender_bank_location',
    'Receiver_bank_location': 'receiver_bank_location',
    'Payment_type': 'payment_type',
    'Is_laundering': 'is_laundering',
    'Laundering_type': 'laundering_type'
}

SAML_D_INTERIM_COLUMNS: tuple[str, ...] = (
    'transaction_id',
    'timestamp',
    'sender_account',
    'receiver_account',
    'amount',
    'payment_currency',
    'received_currency',
    'sender_bank_location',
    'receiver_bank_location',
    'payment_type',
    'is_laundering',
    'laundering_type'
)

SAML_D_INTERIM_SCHEMA = {
    'transaction_id': pl.UInt64,
    'timestamp': pl.Datetime('us'),
    'sender_account': pl.String,
    'receiver_account': pl.String,
    'amount': pl.Float64,
    'payment_currency': pl.String,
    'received_currency': pl.String,
    'sender_bank_location': pl.String,
    'receiver_bank_location': pl.String,
    'payment_type': pl.String,
    'is_laundering': pl.Int8,
    'laundering_type': pl.String
}