import os
import pandas as pd
import wi_cleaning_utils as u

INPUT = 'wi_data/wi_records_anonymized.csv'
OUTPUT = 'wi_data/wi_records_cleaned_raw.csv'
COLUMNS_FILE = 'wi_columns.json'
LOG_FILE = 'wi_cleaning_log_raw.txt'

KEY = 'provider_id'  # provider_location is renamed to provider_id in finalize
TARGET = 'qr_rating'  # youngstar_star_rating is renamed to qr_rating in finalize

# Only these youngstar_star_rating scores are valid; rows whose rating is
# anything else (0, 6, 2.5, 'Not Rated', NaN, ...) are dropped by finalize().
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime column families retained by finalize().
DYNAMIC_PREFIXES = ('service_', 'language_', 'philosophy_', 'care_',
                    'violations_', 'monitoring_', 'enforcement_')


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
    else is untouched. Kept for parity with the GA pipeline; no-op on WI today."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    create_log_file()
    df = pd.read_csv(INPUT, low_memory=False)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    # Multi-value text -> one text column per discovered item (the phrase where
    # present, else NaN). Schema is discovered from the data, not hardcoded.
    # Language sentences are routed out and decomposed into per-language columns.
    df, _ = u.build_multivalue_columns(
        df, 'youngstar_unique_services', delimiter='|', prefix='service',
        as_bool=False, strip_prefix=r'^this program (provides|offers)\s+',
        skip_regex=r'programming is offered in')
    df, _ = u.build_language_columns(
        df, 'youngstar_unique_services', delimiter='|', prefix='language',
        as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'pr_special_types_of_care', delimiter='.', prefix='care',
        as_bool=False, strip_suffix=r'\s+provided$', skip_values=('none reported',))

    # pr_program_philosophy has no delimiter, so decompose it by keyterm match.
    df, _ = u.build_keyterm_columns(df, 'pr_program_philosophy',
                                    u.PHILOSOPHY_KEYTERMS, 'philosophy',
                                    as_bool=False)

    # Nested JSON -> one text column per discovered key (values joined by ' | ').
    df, _ = u.build_json_key_columns(df, 'regulation_violations_json', 'violations')
    df, _ = u.build_json_key_columns(df, 'regulation_monitoring_json', 'monitoring')
    df, _ = u.build_json_key_columns(df, 'regulation_enforcement_json', 'enforcement')

    # Structured text fields -> their decomposed parts.
    df = u.parse_vacancies(df, log=log)
    df = u.parse_waitlist(df, log=log)

    # regulation_type is already atomic text and is kept as-is via the stable
    # scaffold in wi_columns.json.

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)

    # Collapse embedded newlines/carriage returns in preserved free-text cells so
    # each provider stays on a single physical CSV line. Without this, raw's
    # human-readable text fields inject line breaks that make raw's row count
    # (e.g. via wc -l) diverge from full's, even though the record count matches.
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(
            lambda x: x.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
            if isinstance(x, str) else x)

    df.to_csv(OUTPUT, index=False)