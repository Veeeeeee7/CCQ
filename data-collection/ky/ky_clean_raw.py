import numpy as np
import pandas as pd
import ky_cleaning_utils as u

INPUT = 'ky_data/ky_records_anonymized.csv'
OUTPUT = 'ky_data/ky_records_cleaned_raw.csv'
COLUMNS_FILE = 'ky_columns.json'
LOG_FILE = 'ky_cleaning_log_raw.txt'

KEY = 'provider_id'  # ProviderCLRNumber is renamed to provider_id in finalize
TARGET = 'qr_rating'  # NumberOfStars is renamed to qr_rating in finalize

# 0 means not participating; it and non-numeric/missing ratings are dropped.
VALID_RATINGS = (1, 2, 3, 4, 5)

# Only the JSON-history families are discovered at runtime in raw.
DYNAMIC_PREFIXES = ('inspection_', 'dpoc_', 'ongoing_')


def create_log_file(path=LOG_FILE):
    # Truncate rather than os.remove: remove() is blocked on some mounts.
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a') as f:
        f.write(message + '\n')
    print(message)


def strip_dollar_prefix(series):
    """Remove leading '$' and comma separators from currency strings. Kept
    for parity with the GA/WI pipelines; a no-op on KY."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    create_log_file()
    df = pd.read_csv(INPUT, low_memory=False, dtype={'ProviderCLRNumber': str})
    df = df.replace('', np.nan)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    # LocationZipCode5 is float-like ("42728.0").
    df = df.rename(columns={'LocationZipCode5': 'zip'})
    df = u.clean_zip(df)

    df = u.apply_field_renames(df)

    # Structured JSON -> readable text / plain numbers.
    df, _ = u.parse_hours_of_operation(df, as_bool=False)
    df, _ = u.parse_service_cost(df)

    # History-style JSON -> one text column per key; internal ids excluded.
    df, _ = u.build_json_key_columns(df, 'InspectionHistoryListUpdated',
                                     'inspection', skip_keys=('InspectionId',))
    df, _ = u.build_json_key_columns(df, 'DPOCAgreementsListUpdated', 'dpoc',
                                     skip_keys=('docId', 'DocumentID'))
    df, _ = u.build_json_key_columns(df, 'OngoingProcessListUpdated', 'ongoing')

    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').astype('Int64')

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
    log(f'Wrote {len(df)} rows, {len(df.columns)} columns -> {OUTPUT}')
