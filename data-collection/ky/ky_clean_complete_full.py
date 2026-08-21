import numpy as np
import pandas as pd
import ky_cleaning_utils as u

INPUT = 'ky_data/ky_records_anonymized.csv'
OUTPUT = 'ky_data/ky_records_cleaned_complete_full.csv'
COLUMNS_FILE = 'ky_columns.json'

KEY = 'provider_id'  # ProviderCLRNumber is renamed to provider_id in finalize
TARGET = 'qr_rating'  # NumberOfStars is renamed to qr_rating in finalize

# Discovered-at-runtime boolean families retained by finalize(). The
# JSON-history fields collapse to fixed-name count/flag columns for full
# (listed directly in ky_columns.json), so only the one-hots need this.
DYNAMIC_PREFIXES = ('providertype_', 'providerstatus_', 'county_')


if __name__ == "__main__":
    df = pd.read_csv(INPUT, low_memory=False, dtype={'ProviderCLRNumber': str})
    df = df.replace('', np.nan)

    # Native kynect field names -> the project's snake_case schema.
    df = u.apply_field_renames(df)

    # Nested JSON -> numeric/boolean.
    df, _ = u.parse_hours_of_operation(df, as_bool=True)
    df, _ = u.parse_service_cost(df)
    df = u.inspection_counts(df)
    df = u.dpoc_counts(df)
    df = u.ongoing_process_counts(df)

    # Single-value categoricals -> one-hot booleans over discovered values.
    df, _ = u.build_categorical_onehot(df, 'provider_type', 'providertype')
    df, _ = u.build_categorical_onehot(df, 'provider_status', 'providerstatus')
    df, _ = u.build_categorical_onehot(df, 'county', 'county')

    # Native Y/N flags -> real booleans.
    df = u.convert_flags_to_bool(df)

    # capacity is inherently numeric; coerce cleanly.
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').astype('Int64')

    # valid_target_values is omitted, so finalize() keeps every row, including
    # rows whose qr_rating is 0 (not participating/opted out), non-numeric, or
    # missing.
    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES)
    df.to_csv(OUTPUT, index=False)
