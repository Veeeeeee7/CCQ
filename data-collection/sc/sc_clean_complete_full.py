import pandas as pd
import sc_cleaning_utils as u

INPUT = 'sc_data/sc_records_anonymized.csv'
OUTPUT = 'sc_data/sc_records_cleaned_complete_full.csv'
COLUMNS_FILE = 'sc_columns.json'

KEY = 'provider_id'   # permit_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # abc_level is renamed to qr_rating in finalize

# Discovered-at-runtime boolean families retained by finalize(). Same full
# scaffold as sc_clean_full.py: complete == full preprocessing, only the target
# row filter differs (pending / never-rated providers are kept here).
DYNAMIC_PREFIXES = ('permittype_', 'facilitytype_', 'county_')


if __name__ == "__main__":
    # dtype=str + keep_default_na=False: permit_number is numeric-looking and
    # must never round-trip through a float, and blank cells must stay '' rather
    # than becoming NaN (the exempt providers are identified by a blank permit).
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False, low_memory=False)

    df = u.normalize_source_columns(df)
    df = u.synthesize_exempt_ids(df)

    # Reads the *letter* abc_level, so it must run before map_rating() replaces
    # it with the 1-5 ordinal. (add_rating_status() used to run here too; it was
    # removed as target leakage -- see sc_cleaning_utils.LEAKAGE_COLS.)
    df = u.recode_facility_type(df)   # collapses the leaky exempt codes

    df['abc_level'] = u.map_rating(df['abc_level'])

    # Text date -> numeric recency + an explicit "was it ever inspected" flag.
    df = u.parse_inspection_date(df)

    # Categoricals -> one-hot booleans over their discovered values.
    df, _ = u.build_categorical_onehot(df, 'permit_type', 'permittype')
    df, _ = u.build_categorical_onehot(df, 'facility_type_code', 'facilitytype')
    df, _ = u.build_categorical_onehot(df, 'county', 'county')

    # valid_target_values is omitted, so finalize() keeps every row, including
    # the pending ('P') and never-rated providers, whose qr_rating is <NA>.
    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES)
    df.to_csv(OUTPUT, index=False)
