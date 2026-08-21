import os
import mt_cleaning_utils as u

INPUT = 'mt_data/mt_records_anonymized.csv'
OUTPUT = 'mt_data/mt_records_cleaned_full.csv'
COLUMNS_FILE = 'mt_columns.json'
LOG_FILE = 'mt_cleaning_log_full.txt'

KEY = 'provider_id'   # provider_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # star_level is renamed to qr_rating in finalize

# Only these STARS levels are valid; rows whose rating is anything else
# ("Pre-Star" = accepted but not yet rated, NaN, ...) are dropped by finalize().
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime boolean families retained by finalize().
DYNAMIC_PREFIXES = ('ptype_', 'regtype_', 'county_', 'region_', 'name_')


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

    # Text -> numeric/boolean scalars. Their text byproducts (city, county,
    # program_type, provider_type, ccrr_region, zip_code, program_name,
    # provider_id_source) are NOT in the full scaffold, so finalize() drops
    # them, leaving the full set numeric/boolean only.
    df = u.add_has_license(df)
    df = u.coerce_numeric(df, ['latitude', 'longitude'])

    # "Pre-Star" -> NaN, so the target is numeric before finalize(). The row
    # filter below then removes those rows anyway; the coercion is what keeps
    # the COMPLETE full output numeric too (see mt_clean_complete_full.py).
    df = u.coerce_star_numeric(df, 'star_level')
    log(f'read {len(df)} rows; numeric star_level non-null: '
        f'{int(df["star_level"].notna().sum())}')

    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
    log(f'wrote {OUTPUT}: {df.shape[0]} rows x {df.shape[1]} cols')
