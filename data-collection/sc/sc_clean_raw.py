import os
import pandas as pd
import sc_cleaning_utils as u

INPUT = 'sc_data/sc_records_anonymized.csv'
OUTPUT = 'sc_data/sc_records_cleaned_raw.csv'
COLUMNS_FILE = 'sc_columns.json'
LOG_FILE = 'sc_cleaning_log_raw.txt'

KEY = 'provider_id'   # permit_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # abc_level is renamed to qr_rating in finalize

# Only these ABC Quality levels are valid ratings (C=1 .. A+=5). Rows whose
# rating is anything else -- 'P' (pending) or absent -- arrive as <NA> and are
# dropped by finalize().
VALID_RATINGS = u.VALID_RATINGS

# SC's export is already flat: no delimited multi-value cells, no nested JSON,
# so the raw pipeline discovers no dynamic column families.
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
    """Remove leading '$' and comma separators from currency strings; everything
    else is untouched. Kept for parity with the GA/WI pipelines; no-op on SC."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    create_log_file()
    # dtype=str + keep_default_na=False: permit_number is numeric-looking and
    # must never round-trip through a float, and blank cells must stay '' rather
    # than becoming NaN (the exempt providers are identified by a blank permit).
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False, low_memory=False)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    df = u.normalize_source_columns(df)

    # Exempt providers carry no state permit number; mint a stable surrogate so
    # provider_id is populated and unique (and so they survive the dedup).
    df = u.synthesize_exempt_ids(df, log=log)

    # Reads the *letter* abc_level, so it must run before map_rating() replaces
    # it with the 1-5 ordinal. (add_rating_status() used to run here too; it was
    # removed as target leakage -- see sc_cleaning_utils.LEAKAGE_COLS.)
    df = u.recode_facility_type(df, log=log)   # collapses the leaky exempt codes

    df['abc_level'] = u.map_rating(df['abc_level'])

    # city / zip / county / permit_type / facility_type stay as readable text and
    # are kept via the stable scaffold in sc_columns.json.

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
    log(f'Wrote {OUTPUT}: {df.shape[0]} rows x {df.shape[1]} cols')
