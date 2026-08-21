import os
import pandas as pd
import sc_cleaning_utils as u

INPUT = 'sc_data/sc_records_anonymized.csv'
OUTPUT = 'sc_data/sc_records_cleaned_full.csv'
COLUMNS_FILE = 'sc_columns.json'
LOG_FILE = 'sc_cleaning_log_full.txt'

KEY = 'provider_id'   # permit_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # abc_level is renamed to qr_rating in finalize

# Only these ABC Quality levels are valid ratings (C=1 .. A+=5). Rows whose
# rating is anything else -- 'P' (pending) or absent -- arrive as <NA> and are
# dropped by finalize().
VALID_RATINGS = u.VALID_RATINGS

# Discovered-at-runtime boolean families retained by finalize().
DYNAMIC_PREFIXES = ('permittype_', 'facilitytype_', 'county_')


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a') as f:
        f.write(message + '\n')
    print(message)


if __name__ == "__main__":
    create_log_file()
    # dtype=str + keep_default_na=False: permit_number is numeric-looking and
    # must never round-trip through a float, and blank cells must stay '' rather
    # than becoming NaN (the exempt providers are identified by a blank permit).
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False, low_memory=False)

    df = u.normalize_source_columns(df)
    df = u.synthesize_exempt_ids(df, log=log)

    # Reads the *letter* abc_level, so it must run before map_rating() replaces
    # it with the 1-5 ordinal. (add_rating_status() used to run here too; it was
    # removed as target leakage -- see sc_cleaning_utils.LEAKAGE_COLS.)
    df = u.recode_facility_type(df, log=log)   # collapses the leaky exempt codes

    df['abc_level'] = u.map_rating(df['abc_level'])

    # Text date -> numeric recency + an explicit "was it ever inspected" flag.
    df = u.parse_inspection_date(df, log=log)

    # Categoricals -> one-hot booleans over their discovered values.
    df, _ = u.build_categorical_onehot(df, 'permit_type', 'permittype')
    df, _ = u.build_categorical_onehot(df, 'facility_type_code', 'facilitytype')
    df, _ = u.build_categorical_onehot(df, 'county', 'county')

    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
    log(f'Wrote {OUTPUT}: {df.shape[0]} rows x {df.shape[1]} cols')
