import os
import pandas as pd
import sc_cleaning_utils as u

INPUT = 'sc_data/sc_records_anonymized.csv'
OUTPUT = 'sc_data/sc_records_cleaned_raw.csv'
COLUMNS_FILE = 'sc_columns.json'
LOG_FILE = 'sc_cleaning_log_raw.txt'

KEY = 'provider_id'   # permit_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # abc_level is renamed to qr_rating in finalize

# C=1 .. A+=5. Pending ('P') and unrated rows arrive as <NA> and are dropped.
VALID_RATINGS = u.VALID_RATINGS

# SC's export is flat, so raw discovers no dynamic column families.
DYNAMIC_PREFIXES = ()


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a') as f:
        f.write(message + '\n')
    print(message)


def strip_dollar_prefix(series):
    """Remove leading '$' and comma separators from currency strings. Kept for
    parity with the GA/WI pipelines; no-op on SC."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    create_log_file()
    # permit_number must not round-trip through a float, and the exempt
    # providers' blank permits must stay '' rather than NaN.
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False, low_memory=False)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    df = u.normalize_source_columns(df)

    df = u.synthesize_exempt_ids(df, log=log)

    # Reads the letter abc_level, so it must run before map_rating().
    df = u.recode_facility_type(df, log=log)

    df['abc_level'] = u.map_rating(df['abc_level'])

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
    log(f'Wrote {OUTPUT}: {df.shape[0]} rows x {df.shape[1]} cols')
