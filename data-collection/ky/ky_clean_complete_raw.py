import numpy as np
import pandas as pd
import ky_cleaning_utils as u

INPUT = 'ky_data/ky_records_anonymized.csv'
OUTPUT = 'ky_data/ky_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'ky_columns.json'

KEY = 'provider_id'  # ProviderCLRNumber is renamed to provider_id in finalize
TARGET = 'qr_rating'  # NumberOfStars is renamed to qr_rating in finalize

# Discovered-at-runtime column families retained by finalize(). Only the
# three JSON-history fields need this: provider_type/provider_status/county
# stay as plain named columns in the stable scaffold for raw (no onehot).
DYNAMIC_PREFIXES = ('inspection_', 'dpoc_', 'ongoing_')


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

    # valid_target_values is omitted, so finalize() keeps every row, including
    # rows whose qr_rating is 0 (not participating/opted out), non-numeric, or
    # missing.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df.to_csv(OUTPUT, index=False)
