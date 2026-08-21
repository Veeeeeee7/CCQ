import pandas as pd
import ok_cleaning_utils as u

INPUT = 'ok_data/ok_records_anonymized.csv'
OUTPUT = 'ok_data/ok_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'ok_columns.json'

KEY = 'provider_id'
TARGET = 'qr_rating'  # qr_rating_raw is renamed to qr_rating in finalize

# Discovered-at-runtime column families retained by finalize(). Note:
# "monitoring_" and "complaint_" here only ever match the JSON-key columns
# built below (monitoring_visit_date, complaint_category, ...) -- the native
# monitoring_visits_json / complaint_findings_json source columns are
# stripped out earlier by NON_FEATURE_COLS specifically so they can't also
# match these prefixes (see ok_cleaning_utils.py's comment on that).
DYNAMIC_PREFIXES = ('tag_', 'age_', 'monitoring_', 'complaint_')


if __name__ == "__main__":
    df = pd.read_csv(INPUT, low_memory=False)

    # Multi-value text -> one text column per discovered item (the phrase
    # where present, else NaN). Schema is discovered from the data, not
    # hardcoded. Delimiter matches ok_crawler.py's MULTI_DELIM.
    df, _ = u.build_multivalue_columns(
        df, 'program_tags', delimiter=' | ', prefix='tag', as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'ages_accepted', delimiter=' | ', prefix='age', as_bool=False)

    # Structured text -> decomposed numeric summary (kept in raw too,
    # alongside the native hours_<weekday> text columns).
    df = u.parse_hours(df)
    df = u.summarize_monitoring(df)

    # facility_type is already atomic text and is kept as-is via the stable
    # scaffold in ok_columns.json. total_capacity, has_substantiated_complaints,
    # complaints_since_date, n_monitoring_visits, n_complaint_findings, and the
    # *_section_text NLP columns are also already atomic/native and kept as-is.

    # valid_target_values is omitted, so finalize() keeps every row, including
    # rows whose qr_rating is invalid or missing.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df.to_csv(OUTPUT, index=False)
