import os
import pandas as pd
import ne_cleaning_utils as u

INPUT = 'ne_data/ne_records_anonymized.csv'
OUTPUT = 'ne_data/ne_records_cleaned_complete_full.csv'
COLUMNS_FILE = 'ne_columns.json'
LOG_FILE = 'ne_cleaning_log_complete_full.txt'

KEY = 'provider_id'   # provider_key is renamed to provider_id in finalize
TARGET = 'qr_rating'  # step_rating is renamed to qr_rating in finalize

# Discovered-at-runtime boolean families retained by finalize().
DYNAMIC_PREFIXES = ('age_', 'info_', 'accred_', 'day_', 'ptype_', 'lictype_',
                    'county_')

# Self-reported counts arrive as text.
NUMERIC_COLS = ['capacity', 'full_time_staff', 'part_time_staff']


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
    df = pd.read_csv(INPUT, dtype=str, low_memory=False)
    log(f'loaded {len(df)} facility pages')

    # provider_key: the DHHS license number, or a synthetic STQ<facility_id> for
    # the Head Start / public-school programs that carry no license.
    df = u.build_provider_key(df, log=log)

    # Bonus attributes from the DHHS roster (left join; unmatched keep the rating).
    df = u.mark_licensing_matched(df, log=log)

    # Label-prefixed DHHS blobs -> numeric scalars; their text byproducts are
    # not in the full scaffold, so finalize() drops them.
    df = u.parse_capacity(df)
    df = u.parse_ages(df)
    df = u.parse_hours(df)
    df = u.parse_issue_date(df)
    df, _ = u.parse_days_open(df, as_bool=True)

    # Single-value categoricals -> one-hot booleans over discovered values.
    df, _ = u.build_categorical_onehot(df, 'program_type', 'ptype')
    df, _ = u.build_categorical_onehot(df, 'dhhs_license_type', 'lictype')
    df, _ = u.build_categorical_onehot(df, 'dhhs_county', 'county')

    # Multi-value text -> presence booleans over discovered items.
    df, _ = u.build_multivalue_columns(df, 'age_groups', delimiter=';',
                                       prefix='age', as_bool=True)
    df, _ = u.build_multivalue_columns(df, 'other_program_info', delimiter=';',
                                       prefix='info', as_bool=True)

    # Nested JSON -> numeric count + a presence boolean per accrediting body.
    df = u.accreditation_counts(df)
    df, _ = u.build_accreditation_columns(df, prefix='accred')

    df = u.numeric_columns(df, NUMERIC_COLS)

    df = u.prefer_rated_order(df)
    # valid_target_values is omitted, so finalize() keeps every row, including
    # providers with no Step rating.
    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES)
    # prefer_rated_order() only picks the dedup survivor; restore input order
    # so the raw and full views are row-aligned.
    df = df.sort_index()
    df.to_csv(OUTPUT, index=False)
    log(f'wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
