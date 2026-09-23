import numpy as np
import pandas as pd
import ky_cleaning_utils as u

INPUT = 'ky_data/ky_records_anonymized.csv'
OUTPUT = 'ky_data/ky_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'ky_columns.json'

KEY = 'provider_id'  # ProviderCLRNumber is renamed to provider_id in finalize
TARGET = 'qr_rating'  # NumberOfStars is renamed to qr_rating in finalize

# Only the JSON-history families are discovered at runtime in raw.
DYNAMIC_PREFIXES = ('inspection_', 'dpoc_', 'ongoing_')


def strip_dollar_prefix(series):
    """Remove leading '$' and comma separators from currency strings. Kept
    for parity with the GA/WI pipelines; a no-op on KY."""
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

    # No valid_target_values: every row is kept, whatever its qr_rating.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df.to_csv(OUTPUT, index=False)
