"""md_clean_complete_raw.py — Maryland EXCELS: IDENTICAL text-preserving
feature engineering to md_clean_raw.py, but WITHOUT dropping rows whose
qr_rating is missing/invalid (the "complete" target policy).

The four outputs, on two axes (preprocessing depth x row filtering):

                   valid ratings only        all rows kept (complete)
  raw  (text)      md_cleaned_raw.csv        md_clean_complete_raw.csv
  full (numeric)   md_cleaned_full.csv       md_clean_complete_full.csv

complete_raw and complete_full share the same engineering and the same (no)
row filter, so they are row-aligned with each other. They are NOT row-aligned
with raw/full, which restrict to valid 1-5 ratings only.

Byte-for-byte identical to md_clean_raw.py except: output/log filenames and
one finalize() argument (valid_target_values=None keeps every row, including
EXCELS participants that don't have a published rating yet).

Run:
    python md_clean_complete_raw.py
"""
import os

import pandas as pd

import md_cleaning_utils as u

INPUT = 'md_data/md_records_anonymized.csv'
OUTPUT = 'md_data/md_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'md_columns.json'
LOG_FILE = 'md_cleaning_log_complete_raw.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'

DYNAMIC_PREFIXES = ('achievement_', 'accreditation_')

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
    log(f'[complete_raw] loaded {len(df)} rows from {INPUT}')

    df['qr_rated'] = df['Quality Rating'].notna()

    df = df.rename(columns=RENAME)
    for c in ['scholarship_eligible'] + ENROLLMENT_COLS:
        df[c] = u.yesno_bool(df[c])

    df, ach_cols = u.build_multivalue_columns(
        df, 'Achievements', delimiter='; ', prefix='achievement', as_bool=False)
    df, acc_cols = u.build_multivalue_columns(
        df, 'Accreditations', delimiter='; ', prefix='accreditation', as_bool=False)
    log(f'[complete_raw] discovered {len(ach_cols)} achievement cols, {len(acc_cols)} accreditation cols')

    # which='raw' -> reuse raw's scaffold/discovered prefixes/exclusions/text
    # preservation; valid_target_values=None -> keep unrated/invalid rows too.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=None)

    df.to_csv(OUTPUT, index=False)
    log(f'[complete_raw] wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
