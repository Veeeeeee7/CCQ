import os
import pandas as pd
import ok_cleaning_utils as u

INPUT = 'ok_data/ok_records_anonymized.csv'
OUTPUT = 'ok_data/ok_records_cleaned_raw.csv'
COLUMNS_FILE = 'ok_columns.json'
LOG_FILE = 'ok_cleaning_log_raw.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'  # qr_rating_raw is renamed to qr_rating in finalize

# Only these Star Levels are valid (Checkpoint 0, 2026-07-07): Level 1 is
# automatic on licensing, Levels 2-5 are voluntary, all five are real
# published ratings. Rows whose rating is anything else (out of range,
# non-numeric, or missing) are dropped by finalize().
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime column families retained by finalize(). Note:
# "monitoring_" and "complaint_" here only ever match the JSON-key columns
# built below (monitoring_visit_date, complaint_category, ...) -- the native
# monitoring_visits_json / complaint_findings_json source columns are
# stripped out earlier by NON_FEATURE_COLS specifically so they can't also
# match these prefixes (see ok_cleaning_utils.py's comment on that).
DYNAMIC_PREFIXES = ('tag_', 'age_', 'monitoring_', 'complaint_')


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
    # where present, else NaN). Schema is discovered from the data, not
    # hardcoded. Delimiter matches ok_crawler.py's MULTI_DELIM.
    df, _ = u.build_multivalue_columns(
        df, 'program_tags', delimiter=' | ', prefix='tag', as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'ages_accepted', delimiter=' | ', prefix='age', as_bool=False)

    # Structured text -> decomposed numeric summary (kept in raw too,
    # alongside the native hours_<weekday> text columns, same as CA keeps
    # both business_hours and its derived days_open/earliest_open/... in its
    # raw set).
    df = u.parse_hours(df, log=log)
    df = u.summarize_monitoring(df)

    # facility_type is already atomic text and is kept as-is via the stable
    # scaffold in ok_columns.json. total_capacity, has_substantiated_complaints,
    # complaints_since_date, n_monitoring_visits, n_complaint_findings, and the
    # *_section_text NLP columns are also already atomic/native and kept as-is.

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
