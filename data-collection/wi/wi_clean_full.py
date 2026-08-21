import os
import pandas as pd
import wi_cleaning_utils as u

INPUT = 'wi_data/wi_records_anonymized.csv'
OUTPUT = 'wi_data/wi_records_cleaned_full.csv'
COLUMNS_FILE = 'wi_columns.json'
LOG_FILE = 'wi_cleaning_log_full.txt'

KEY = 'provider_id'  # provider_location is renamed to provider_id in finalize
TARGET = 'qr_rating'  # youngstar_star_rating is renamed to qr_rating in finalize

# Only these youngstar_star_rating scores are valid; rows whose rating is
# anything else (0, 6, 2.5, 'Not Rated', NaN, ...) are dropped by finalize().
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime boolean families retained by finalize().
DYNAMIC_PREFIXES = ('regtype_', 'philosophy_', 'language_', 'service_', 'care_')


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
    df = pd.read_csv(INPUT, low_memory=False)

    # Nested JSON -> numeric counts.
    df = u.json_counts(df)

    # Single-value categoricals -> one-hot booleans over discovered values.
    df, _ = u.build_categorical_onehot(df, 'regulation_type', 'regtype')

    # pr_program_philosophy concatenates multiple philosophies with no delimiter,
    # so match a keyterm vocabulary instead of one-hotting whole-cell values.
    df, _ = u.build_keyterm_columns(df, 'pr_program_philosophy',
                                    u.PHILOSOPHY_KEYTERMS, 'philosophy',
                                    as_bool=True)

    # Multi-value text -> presence booleans over discovered items. Language
    # sentences are routed out of the service columns and decomposed separately.
    df, _ = u.build_multivalue_columns(
        df, 'youngstar_unique_services', delimiter='|', prefix='service',
        as_bool=True, strip_prefix=r'^this program (provides|offers)\s+',
        skip_regex=r'programming is offered in')
    df, _ = u.build_language_columns(
        df, 'youngstar_unique_services', delimiter='|', prefix='language',
        as_bool=True)
    df, _ = u.build_multivalue_columns(
        df, 'pr_special_types_of_care', delimiter='.', prefix='care',
        as_bool=True, strip_suffix=r'\s+provided$', skip_values=('none reported',))

    # Structured text -> numeric/boolean scalars. Their text byproducts
    # (vacancies_age_range, waitlist_last_updated) are NOT in the full scaffold,
    # so finalize() drops them, leaving the full set numeric/boolean only.
    df = u.parse_vacancies(df, log=log)
    df = u.parse_waitlist(df, log=log)

    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)