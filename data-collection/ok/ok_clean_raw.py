import os
import pandas as pd
import ok_cleaning_utils as u

INPUT = 'ok_data/ok_records_anonymized.csv'
OUTPUT = 'ok_data/ok_records_cleaned_raw.csv'
COLUMNS_FILE = 'ok_columns.json'
LOG_FILE = 'ok_cleaning_log_raw.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'  # qr_rating_raw is renamed to qr_rating in finalize

# Valid Star Levels (Level 1 is automatic on licensing, 2-5 are voluntary);
# finalize() drops any other rating.
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime column families retained by finalize(). The native
# monitoring_visits_json / complaint_findings_json columns are dropped earlier
# by NON_FEATURE_COLS so they cannot match these prefixes.
DYNAMIC_PREFIXES = ('tag_', 'age_', 'monitoring_', 'complaint_')

# Decomposed by the privacy block from monitoring_visit_type, but 'Full' is set
# on 2,504 of the 2,507 released rows, so it carries no information. Dropped in
# both raw views; the full view never has it.
CONSTANT_COLS = ['monitoring_visit_type_full']


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

    # Multi-value text -> one text column per discovered item (the phrase
    # where present, else NaN).
    df, _ = u.build_multivalue_columns(
        df, 'program_tags', delimiter=' | ', prefix='tag', as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'ages_accepted', delimiter=' | ', prefix='age', as_bool=False)

    # Structured text -> decomposed numeric summary (kept in raw too,
    # alongside the native hours_<weekday> text columns).
    df = u.parse_hours(df, log=log)
    df = u.summarize_monitoring(df)


    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    df = df.drop(columns=[c for c in CONSTANT_COLS if c in df.columns])
    df[TARGET] = df[TARGET].astype(int)
    df.to_csv(OUTPUT, index=False)
