import mt_cleaning_utils as u

INPUT = 'mt_data/mt_records_anonymized.csv'
OUTPUT = 'mt_data/mt_records_cleaned_complete_full.csv'
COLUMNS_FILE = 'mt_columns.json'

KEY = 'provider_id'   # provider_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # star_level is renamed to qr_rating in finalize

# Discovered-at-runtime boolean families retained by finalize(). Same full
# scaffold as mt_clean_full.py: complete == full preprocessing, only the target
# row filter differs (invalid/missing ratings are kept here).
DYNAMIC_PREFIXES = ('ptype_', 'regtype_', 'county_', 'region_', 'name_')


if __name__ == "__main__":
    # Reads as str (IDs must never be coerced) and hard-fails on a stale
    # records file.
    df = u.read_records(INPUT)

    # Single-value categoricals -> one-hot over their discovered value spaces.
    # program_type is the STARS program type (Center/Group/Family); provider_type
    # is the licensing registration type, which is the closer analogue of WI's
    # regulation_type, hence the regtype_ prefix.
    df, _ = u.build_categorical_onehot(df, 'program_type', 'ptype')
    df, _ = u.build_categorical_onehot(df, 'provider_type', 'regtype')
    df, _ = u.build_categorical_onehot(df, 'county', 'county')
    df, _ = u.build_categorical_onehot(df, 'ccrr_region', 'region')

    # program_name concatenates descriptors with no delimiter, so match a
    # keyterm vocabulary instead of one-hotting whole-cell values.
    df = u.adopt_keyterm_columns(df, 'name', as_bool=True)

    # Text -> numeric/boolean scalars. Their text byproducts are NOT in the full
    # scaffold, so finalize() drops them, leaving the full set numeric/boolean.
    df = u.add_has_license(df)
    df = u.coerce_numeric(df, ['latitude', 'longitude'])

    # Coercing here is what keeps this output numeric: no row filter runs below,
    # so the "Pre-Star" programs survive, and they must survive as NaN rather
    # than as the literal string "Pre-Star".
    df = u.coerce_star_numeric(df, 'star_level')

    # valid_target_values is omitted, so finalize() keeps every row, including
    # the "Pre-Star" programs (accepted into STARS but not yet rated), whose
    # qr_rating is NaN here.
    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES)
    df.to_csv(OUTPUT, index=False)
