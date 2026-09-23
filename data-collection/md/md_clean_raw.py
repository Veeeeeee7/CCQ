"""md_clean_raw.py — Maryland EXCELS: text-preserving cleaning, valid ratings only.

Row-aligned with md_clean_full.py.

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

KEY = 'provider_id'
TARGET = 'qr_rating'

DYNAMIC_PREFIXES = ('achievement_', 'accreditation_')

RENAME = {
    'Program Type': 'program_type',
    'Scholarship Eligible': 'scholarship_eligible',
    'Enrollment Availability': 'enrollment_availability',
    # Licensed age ranges and capacity: the program's licence scope. The
    # '6 weeks-17 mos' ... '5 yrs-15 yrs' columns are openings per age band,
    # not ages served, and are deliberately not renamed or kept.
    'Licensed 6 weeks-17 mos': 'age_6wk_17mo',
    'Licensed 18 mos-23 mos': 'age_18_23mo',
    'Licensed 0 mos-23 mos': 'age_0_23mo',
    'Licensed 2 years': 'age_2yr',
    'Licensed 3 years': 'age_3yr',
    'Licensed 4 years': 'age_4yr',
    'Licensed 5 yrs preschool': 'age_5yr_preschool',
    'Licensed 5 yrs-15 yrs': 'age_5_15yr',
    'License Capacity': 'licensed_capacity',
}

# Present in every stage-1 file.
YESNO_COLS = ['scholarship_eligible', 'enrollment_availability']

# Present only once the licensed age/capacity columns have been merged.
# Values are 'Yes'/'No', so a blank means unpublished or unmatched, not "no".
AGE_COLS = [
    'age_6wk_17mo', 'age_18_23mo', 'age_0_23mo', 'age_2yr', 'age_3yr',
    'age_4yr', 'age_5yr_preschool', 'age_5_15yr',
]
CAPACITY_COL = 'licensed_capacity'


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

    # Whether the program has a published rating yet; native column name.
    df['qr_rated'] = df['Quality Rating'].notna()

    df = df.rename(columns=RENAME)
    for c in YESNO_COLS:
        df[c] = u.yesno_bool(df[c])

    # Runs without the licensed age/capacity columns, but a half-merged file
    # is a bug.
    present = [c for c in AGE_COLS if c in df.columns]
    if present and len(present) != len(AGE_COLS):
        log(f'[raw] WARNING: {len(present)} of {len(AGE_COLS)} licensed age '
            f'columns present in {INPUT} -- it looks half-merged')
    for c in present:
        df[c] = u.yesno_bool(df[c])
    if CAPACITY_COL in df.columns:
        # Blank -> <NA>: no capacity published (every Public Prekindergarten
        # row, plus a few centres).
        df[CAPACITY_COL] = pd.to_numeric(df[CAPACITY_COL],
                                         errors='coerce').astype('Int64')
    log(f'[raw] licensed age columns: {len(present)}/{len(AGE_COLS)}; '
        f'licensed_capacity: {"yes" if CAPACITY_COL in df.columns else "no"}')

    df, ach_cols = u.build_multivalue_columns(
        df, 'Achievements', delimiter='; ', prefix='achievement', as_bool=False)
    df, acc_cols = u.build_multivalue_columns(
        df, 'Accreditations', delimiter='; ', prefix='accreditation', as_bool=False)
    log(f'[raw] discovered {len(ach_cols)} achievement cols, {len(acc_cols)} accreditation cols')

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=u.VALID_RATINGS)

    df.to_csv(OUTPUT, index=False)
    log(f'[raw] wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
