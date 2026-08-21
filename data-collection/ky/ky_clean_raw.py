import numpy as np
import pandas as pd
import ky_cleaning_utils as u

INPUT = 'ky_data/ky_records_anonymized.csv'
OUTPUT = 'ky_data/ky_records_cleaned_raw.csv'
COLUMNS_FILE = 'ky_columns.json'
LOG_FILE = 'ky_cleaning_log_raw.txt'

KEY = 'provider_id'  # ProviderCLRNumber is renamed to provider_id in finalize
TARGET = 'qr_rating'  # NumberOfStars is renamed to qr_rating in finalize

# Only these NumberOfStars scores are valid; 0 is kynect's code for
# not-participating/opted-out (see ky_cleaning_utils.py docstring on
# ID_RENAME/TARGET_RENAME), so it -- along with anything non-numeric or
# missing -- is dropped by finalize() here and only survives in the
# *_complete_* outputs.
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime column families retained by finalize(). Only the
# three JSON-history fields need this: provider_type/provider_status/county
# stay as plain named columns in the stable scaffold for raw (no onehot).
DYNAMIC_PREFIXES = ('inspection_', 'dpoc_', 'ongoing_')


def create_log_file(path=LOG_FILE):
    # open(..., 'w') already truncates/creates; skip os.remove -- on some
    # mounted filesystems remove() is blocked even though write/truncate
    # isn't.
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a') as f:
        f.write(message + '\n')
    print(message)


def strip_dollar_prefix(series):
    """Remove leading '$' and comma separators from currency strings; everything
    else is untouched. Kept for parity with the GA/WI pipelines; no-op on KY
    today (ServiceCostList's costs are plain JSON numbers, not $-strings)."""
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

    # LocationZipCode5 round-trips through the crawler as a float-like string
    # ("42728.0"); this must run before the rename below (still native name).
    df = df.rename(columns={'LocationZipCode5': 'zip'})
    df = u.clean_zip(df)

    # Remaining native kynect field names -> the project's snake_case schema.
    df = u.apply_field_renames(df)

    # Structured JSON -> their decomposed parts (readable text/plain numbers).
    df, _ = u.parse_hours_of_operation(df, as_bool=False)
    df, _ = u.parse_service_cost(df)

    # History-style JSON -> one text column per discovered key (values joined
    # by ' | '). InspectionId/docId/DocumentID are internal ids, not useful
    # as readable text, so they're excluded.
    df, _ = u.build_json_key_columns(df, 'InspectionHistoryListUpdated',
                                     'inspection', skip_keys=('InspectionId',))
    df, _ = u.build_json_key_columns(df, 'DPOCAgreementsListUpdated', 'dpoc',
                                     skip_keys=('docId', 'DocumentID'))
    df, _ = u.build_json_key_columns(df, 'OngoingProcessListUpdated', 'ongoing')

    # capacity is inherently numeric; coerce cleanly.
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').astype('Int64')

    # provider_type/provider_status/county/the Y-N flag columns are already
    # atomic text and are kept as-is via the stable scaffold in ky_columns.json.

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
    log(f'Wrote {len(df)} rows, {len(df.columns)} columns -> {OUTPUT}')
