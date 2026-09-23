import pandas as pd
import sc_cleaning_utils as u

INPUT = 'sc_data/sc_records_anonymized.csv'
OUTPUT = 'sc_data/sc_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'sc_columns.json'

KEY = 'provider_id'   # permit_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # abc_level is renamed to qr_rating in finalize

# SC's export is flat, so raw discovers no dynamic column families.
DYNAMIC_PREFIXES = ()


def strip_dollar_prefix(series):
    """Remove leading '$' and comma separators from currency strings. Kept for
    parity with the GA/WI pipelines; no-op on SC."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    # permit_number must not round-trip through a float, and the exempt
    # providers' blank permits must stay '' rather than NaN.
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False, low_memory=False)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    df = u.normalize_source_columns(df)

    df = u.synthesize_exempt_ids(df)

    # Reads the letter abc_level, so it must run before map_rating().
    df = u.recode_facility_type(df)

    df['abc_level'] = u.map_rating(df['abc_level'])

    # No valid_target_values: pending and unrated providers (<NA>) are kept.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df.to_csv(OUTPUT, index=False)
