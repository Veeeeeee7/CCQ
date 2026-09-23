import os
import pandas as pd
import ok_cleaning_utils as u

INPUT = 'ok_data/ok_records_anonymized.csv'
OUTPUT = 'ok_data/ok_records_cleaned_full.csv'
COLUMNS_FILE = 'ok_columns.json'
LOG_FILE = 'ok_cleaning_log_full.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'  # qr_rating_raw is renamed to qr_rating in finalize

# Valid Star Levels (Level 1 is automatic on licensing, 2-5 are voluntary);
# finalize() drops any other rating.
VALID_RATINGS = (1, 2, 3, 4, 5)

# Discovered-at-runtime boolean families retained by finalize().
DYNAMIC_PREFIXES = ('factype_', 'tag_', 'age_')


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

    # Single-value categorical -> one-hot booleans over discovered values.
    df, _ = u.build_categorical_onehot(df, 'facility_type', 'factype')

    # Multi-value text -> presence booleans over discovered items.
    df, _ = u.build_multivalue_columns(
        df, 'program_tags', delimiter=' | ', prefix='tag', as_bool=True)
    df, _ = u.build_multivalue_columns(
        df, 'ages_accepted', delimiter=' | ', prefix='age', as_bool=True)

    # Structured text -> numeric scalars.
    df = u.parse_hours(df, log=log)
    df = u.summarize_monitoring(df)

    # Native scalar columns -> proper numeric/boolean dtypes.
    df['total_capacity'] = pd.to_numeric(df['total_capacity'], errors='coerce').astype('Int64')
    df['n_monitoring_visits'] = pd.to_numeric(df['n_monitoring_visits'], errors='coerce').astype('Int64')
    df['n_complaint_findings'] = pd.to_numeric(df['n_complaint_findings'], errors='coerce').astype('Int64')
    df['has_substantiated_complaints'] = u.to_bool(df['has_substantiated_complaints'])

    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    valid_target_values=VALID_RATINGS)
    df[TARGET] = df[TARGET].astype(int)
    df.to_csv(OUTPUT, index=False)
