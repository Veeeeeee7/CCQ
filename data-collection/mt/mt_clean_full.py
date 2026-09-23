import os
import mt_cleaning_utils as u

INPUT = 'mt_data/mt_records_anonymized.csv'
OUTPUT = 'mt_data/mt_records_cleaned_full.csv'
COLUMNS_FILE = 'mt_columns.json'
LOG_FILE = 'mt_cleaning_log_full.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'

# "Pre-Star" (accepted but not yet rated) is not a valid rating.
VALID_RATINGS = (1, 2, 3, 4, 5)

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
    log(f'read {len(df)} rows; numeric star_level non-null: '
        f'{int(df["star_level"].notna().sum())}')

    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    valid_target_values=VALID_RATINGS)
    df[TARGET] = df[TARGET].astype(int)
    df.to_csv(OUTPUT, index=False)
    log(f'wrote {OUTPUT}: {df.shape[0]} rows x {df.shape[1]} cols')
