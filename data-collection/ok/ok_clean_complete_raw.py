import os

import pandas as pd
import ok_cleaning_utils as u

INPUT = 'ok_data/ok_records_anonymized.csv'
OUTPUT = 'ok_data/ok_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'ok_columns.json'
LOG_FILE = 'ok_cleaning_log_complete_raw.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'  # qr_rating_raw is renamed to qr_rating in finalize

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
        try:
            os.remove(path)
        except OSError:
            pass  # some mounts disallow unlink; the 'w' below truncates anyway
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


    # valid_target_values is omitted, so finalize() keeps every row, including
    # rows whose qr_rating is invalid or missing.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df = df.drop(columns=[c for c in CONSTANT_COLS if c in df.columns])
    # Nullable Int64: this view keeps unrated rows, which plain int cannot
    # hold; without the cast every rating is written '4.0'.
    _numeric = pd.to_numeric(df[TARGET], errors='coerce')
    assert (_numeric.isna() == df[TARGET].isna()).all(), \
        'a non-numeric rating would be lost by the Int64 cast'
    df[TARGET] = _numeric.astype('Int64')
    df.to_csv(OUTPUT, index=False)
    log(f'wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
