"""md_clean_complete_full.py — Maryland EXCELS: IDENTICAL numeric/boolean
feature engineering to md_clean_full.py, but WITHOUT dropping rows whose
qr_rating is missing/invalid (the "complete" target policy).

See md_clean_complete_raw.py's docstring for the full 2x2 explanation.
Byte-for-byte identical to md_clean_full.py except: output/log filenames and
one finalize() argument (valid_target_values=None keeps every row, including
EXCELS participants that don't have a published rating yet).

Run:
    python md_clean_complete_full.py
"""
import os

import pandas as pd

import md_cleaning_utils as u

INPUT = 'md_data/md_records_anonymized.csv'
OUTPUT = 'md_data/md_records_cleaned_complete_full.csv'
COLUMNS_FILE = 'md_columns.json'
LOG_FILE = 'md_cleaning_log_complete_full.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'

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
    log(f'[complete_full] loaded {len(df)} rows from {INPUT}')

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
    log(f'[complete_full] discovered {len(pt_cols)} program_type cols, '
        f'{len(ach_cols)} achievement cols, {len(acc_cols)} accreditation cols')

    # which='full' -> reuse full's scaffold/discovered prefixes/exclusions/
    # strict numeric-boolean shape; valid_target_values=None -> keep unrated
    # rows too.
    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=False, valid_target_values=None)

    bad = [c for c in df.columns
           if c != 'provider_id' and not pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c]) and df[c].dtype != 'boolean']
    if bad:
        log(f'[complete_full] WARNING non-numeric columns survived: {bad}')

    df.to_csv(OUTPUT, index=False)
    log(f'[complete_full] wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
