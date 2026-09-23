import os
import pandas as pd
import ne_cleaning_utils as u

INPUT = 'ne_data/ne_records_anonymized.csv'
OUTPUT = 'ne_data/ne_records_cleaned_raw.csv'
COLUMNS_FILE = 'ne_columns.json'
LOG_FILE = 'ne_cleaning_log_raw.txt'

KEY = 'provider_id'   # provider_key is renamed to provider_id in finalize
TARGET = 'qr_rating'  # step_rating is renamed to qr_rating in finalize

# Valid Step Up to Quality steps; finalize() drops any other rating.
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime column families retained by finalize().
DYNAMIC_PREFIXES = ('age_', 'info_', 'accreditation_', 'day_')

# Self-reported counts arrive as text.
NUMERIC_COLS = ['capacity', 'full_time_staff', 'part_time_staff', 'zip_code']


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
    else is untouched. Kept for parity with the GA/WI pipelines; no-op on NE."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    create_log_file()
    df = pd.read_csv(INPUT, dtype=str, low_memory=False)
    log(f'loaded {len(df)} facility pages')

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    # provider_key: the DHHS license number, or a synthetic STQ<facility_id> for
    # the Head Start / public-school programs that carry no license.
    df = u.build_provider_key(df, log=log)

    # Bonus attributes from the DHHS roster (left join; unmatched keep the rating).
    df = u.mark_licensing_matched(df, log=log)

    # Label-prefixed DHHS blobs -> readable text plus their decomposed parts.
    df = u.parse_capacity(df)
    df = u.parse_ages(df)
    df = u.parse_hours(df)
    df = u.parse_issue_date(df)
    df, _ = u.parse_days_open(df, as_bool=False)

    # Multi-value text -> one text column per discovered item (the phrase where
    # present, else NaN).
    df, _ = u.build_multivalue_columns(df, 'age_groups', delimiter=';',
                                       prefix='age', as_bool=False)
    df, _ = u.build_multivalue_columns(df, 'other_program_info', delimiter=';',
                                       prefix='info', as_bool=False)

    # Nested JSON -> one text column per discovered key (values joined by ' | ').
    df, _ = u.build_json_key_columns(df, 'accreditations', 'accreditation')
    df = u.accreditation_counts(df)

    df = u.numeric_columns(df, NUMERIC_COLS)

    # `city` and `zip_code` are in the raw scaffold only so they reach the
    # geography stage, which drops them: NE has no region, so `county` is the
    # single geographic field that ships.

    df = u.prefer_rated_order(df)
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    # prefer_rated_order() only picks the dedup survivor; restore input order
    # so the raw and full views are row-aligned.
    df = df.sort_index()
    df.to_csv(OUTPUT, index=False)
    log(f'wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
