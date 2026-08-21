"""md_clean_full.py — Maryland EXCELS: strictly numeric/boolean cleaning,
valid ratings only, for classical/tabular ML.

Same early steps and row filtering as md_clean_raw.py (row-aligned with it);
multi-value fields become presence booleans and program_type becomes one-hot
instead of staying free text.

Run:
    python md_clean_full.py
"""
import os

import pandas as pd

import md_cleaning_utils as u

INPUT = 'md_data/md_records_anonymized.csv'
OUTPUT = 'md_data/md_records_cleaned_full.csv'
COLUMNS_FILE = 'md_columns.json'
LOG_FILE = 'md_cleaning_log_full.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'

# Discovered-at-runtime column families retained by finalize(). program_type_*
# only exists in full (raw keeps program_type as a single text column).
DYNAMIC_PREFIXES = ('achievement_', 'accreditation_', 'program_type_')

RENAME = {
    'Program Type': 'program_type',
    'Scholarship Eligible': 'scholarship_eligible',
    'Enrollment Availability': 'enrollment_availability',
    '6 weeks-17 mos': 'enrollment_age_6wk_17mo',
    '18 mos-23 mos': 'enrollment_age_18_23mo',
    '0 mos-23 mos': 'enrollment_age_0_23mo',
    '2 years': 'enrollment_age_2yr',
    '3 years': 'enrollment_age_3yr',
    '4 years': 'enrollment_age_4yr',
    '5 yrs preschool': 'enrollment_age_5yr_preschool',
    '5 yrs-15 yrs': 'enrollment_age_5_15yr',
}

ENROLLMENT_COLS = [
    'enrollment_availability', 'enrollment_age_6wk_17mo',
    'enrollment_age_18_23mo', 'enrollment_age_0_23mo', 'enrollment_age_2yr',
    'enrollment_age_3yr', 'enrollment_age_4yr', 'enrollment_age_5yr_preschool',
    'enrollment_age_5_15yr',
]


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a') as f:
        f.write(message + '\n')
    print(message)


if __name__ == '__main__':
    create_log_file()
    df = pd.read_csv(INPUT, dtype=str, low_memory=False)  # preserve leading zeros (Program ID)
    log(f'[full] loaded {len(df)} rows from {INPUT}')

    df['qr_rated'] = df['Quality Rating'].notna()

    df = df.rename(columns=RENAME)
    for c in ['scholarship_eligible'] + ENROLLMENT_COLS:
        df[c] = u.yesno_bool(df[c])

    df, pt_cols = u.build_categorical_onehot(df, 'program_type', 'program_type')
    df = df.drop(columns=['program_type'])

    df, ach_cols = u.build_multivalue_columns(
        df, 'Achievements', delimiter='; ', prefix='achievement', as_bool=True)
    df, acc_cols = u.build_multivalue_columns(
        df, 'Accreditations', delimiter='; ', prefix='accreditation', as_bool=True)
    log(f'[full] discovered {len(pt_cols)} program_type cols, '
        f'{len(ach_cols)} achievement cols, {len(acc_cols)} accreditation cols')

    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=False, valid_target_values=u.VALID_RATINGS)

    # sanity: every non-id column must be numeric/boolean
    bad = [c for c in df.columns
           if c != 'provider_id' and not pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c]) and df[c].dtype != 'boolean']
    if bad:
        log(f'[full] WARNING non-numeric columns survived: {bad}')

    df.to_csv(OUTPUT, index=False)
    log(f'[full] wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
