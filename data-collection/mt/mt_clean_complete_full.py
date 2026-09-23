import pandas as pd
import mt_cleaning_utils as u

INPUT = 'mt_data/mt_records_anonymized.csv'
OUTPUT = 'mt_data/mt_records_cleaned_complete_full.csv'
COLUMNS_FILE = 'mt_columns.json'

KEY = 'provider_id'
TARGET = 'qr_rating'

DYNAMIC_PREFIXES = ('ptype_', 'regtype_', 'county_', 'region_', 'name_')


if __name__ == "__main__":
    df = u.read_records(INPUT)

    # provider_type is the licensing registration type, hence regtype_.
    df, _ = u.build_categorical_onehot(df, 'program_type', 'ptype')
    df, _ = u.build_categorical_onehot(df, 'provider_type', 'regtype')
    df, _ = u.build_categorical_onehot(df, 'county', 'county')
    df, _ = u.build_categorical_onehot(df, 'ccrr_region', 'region')

    df = u.adopt_keyterm_columns(df, 'name', as_bool=True)

    # Text byproducts are not in the full scaffold, so finalize() drops them.
    df = u.add_has_license(df)
    df = u.coerce_numeric(df, ['latitude', 'longitude'])

    # "Pre-Star" -> NaN, so the target is numeric before finalize().
    df = u.coerce_star_numeric(df, 'star_level')

    # No valid_target_values: every row is kept, "Pre-Star" as NaN.
    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES)
    # Nullable Int64: plain int cannot hold the unrated rows' <NA>, and without
    # the cast every rating is written '4.0'.
    _numeric = pd.to_numeric(df[TARGET], errors='coerce')
    assert (_numeric.isna() == df[TARGET].isna()).all(), \
        'a non-numeric rating would be lost by the Int64 cast'
    df[TARGET] = _numeric.astype('Int64')
    df.to_csv(OUTPUT, index=False)
