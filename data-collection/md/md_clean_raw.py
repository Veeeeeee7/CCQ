"""md_clean_raw.py — Maryland EXCELS: text-preserving cleaning, valid ratings only.

Same early steps and row filtering as md_clean_full.py (row-aligned with it);
the per-field handling just keeps text instead of exploding to booleans-only.

Run:
    python md_clean_raw.py
"""
import os

import pandas as pd

import md_cleaning_utils as u

INPUT = 'md_data/md_records_anonymized.csv'
OUTPUT = 'md_data/md_records_cleaned_raw.csv'
COLUMNS_FILE = 'md_columns.json'
LOG_FILE = 'md_cleaning_log_raw.txt'

KEY = 'provider_id'    # Program ID is renamed to provider_id in finalize
TARGET = 'qr_rating'   # Quality Rating is renamed to qr_rating in finalize

# Discovered-at-runtime column families retained by finalize().
DYNAMIC_PREFIXES = ('achievement_', 'accreditation_')

# Source column -> canonical name. (Program ID / Quality Rating are handled
# separately by finalize()'s ID_RENAME/TARGET_RENAME.)
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
    log(f'[raw] loaded {len(df)} rows from {INPUT}')

    # qr_rated: whether this EXCELS participant has a published rating yet.
    # Computed before finalize()'s rename, from the still-native column name.
    df['qr_rated'] = df['Quality Rating'].notna()

    df = df.rename(columns=RENAME)
    for c in ['scholarship_eligible'] + ENROLLMENT_COLS:
        df[c] = u.yesno_bool(df[c])

    # Multi-value text -> one text column per discovered item (raw: the
    # phrase where present, else NaN). Vocabulary discovered from the data,
    # not hardcoded (see md_cleaning_utils.py docstring).
    df, ach_cols = u.build_multivalue_columns(
        df, 'Achievements', delimiter='; ', prefix='achievement', as_bool=False)
    df, acc_cols = u.build_multivalue_columns(
        df, 'Accreditations', delimiter='; ', prefix='accreditation', as_bool=False)
    log(f'[raw] discovered {len(ach_cols)} achievement cols, {len(acc_cols)} accreditation cols')

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=u.VALID_RATINGS)

    df.to_csv(OUTPUT, index=False)
    log(f'[raw] wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
