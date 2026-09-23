import os

import pandas as pd
import wa_cleaning_utils as u

INPUT = 'wa_data/wa_records_anonymized.csv'
OUTPUT = 'wa_data/wa_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'wa_columns.json'
LOG_FILE = 'wa_cleaning_log_complete_raw.txt'

KEY = 'provider_id'   # Salesforce `Id` is renamed to provider_id in finalize
TARGET = 'qr_rating'  # Early_Achiever_Status_Internal__c -> qr_rating in finalize

# Discovered-at-runtime column families retained by finalize().
DYNAMIC_PREFIXES = ('agegroup_', 'slotavail_', 'spec_', 'language_', 'langinstr_',
                    'complaints_', 'inspections_', 'licensehist_', 'contacts_')

# Near-constant in the released view, so wa_clean_raw.py drops them; dropped
# here too so the two raw views get the same treatment.
CONSTANT_COLS = ['licensehist_license_type_non_expiring',
                 'licensehist_regulation_type_dcyf_licensed']

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



def strip_dollar_prefix(series):
    """Remove leading '$' and comma separators from currency strings. Kept for
    parity with the GA pipeline; no-op on WA."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    create_log_file()
    df = pd.read_csv(INPUT, low_memory=False)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    # The target mixes levels with non-rating statuses; reduce it to a number.
    df = u.normalize_rating(df)

    # Renamed up front so the parsers below read the tidy names.
    df = df.rename(columns=u.FEATURE_RENAME)

    # Structured text fields -> their decomposed parts, text preserved.
    df = u.parse_ages_served(df)
    df = u.parse_contact_blob(df)
    df = u.parse_location(df)
    df = u.parse_hours(df)
    df = u.parse_license_dates(df)

    # Multi-value text -> one text column per discovered item (the phrase where
    # present, else NaN).
    df, _ = u.build_multivalue_columns(
        df, 'age_groups_served', delimiter=';', prefix='agegroup', as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'slot_availability', delimiter=';', prefix='slotavail', as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'ea_specialization', delimiter=';', prefix='spec', as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'languages_spoken', delimiter=';', prefix='language', as_bool=False)
    df, _ = u.build_multivalue_columns(
        df, 'languages_of_instruction', delimiter=';', prefix='langinstr',
        as_bool=False)

    # Nested JSON -> one text column per discovered key (values joined by ' | ').
    df, _ = u.build_json_key_columns(df, 'complaints_json', 'complaints')
    df, _ = u.build_json_key_columns(df, 'inspections_json', 'inspections')
    df, _ = u.build_json_key_columns(df, 'license_history_json', 'licensehist')
    df, _ = u.build_json_key_columns(df, 'contacts_json', 'contacts')

    df = u.json_counts(df)

    # A blank *_count means the detail page was never seen: unknown, not zero.
    df = u.mask_unknown_counts(df, log=log)

    # No valid_target_values: rows with a missing qr_rating are kept.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df = df.drop(columns=[c for c in CONSTANT_COLS if c in df.columns])

    # Uses post-rename names, so it has to run after finalize().
    df = u.cast_int_columns(df, log=log)
    df.to_csv(OUTPUT, index=False)
    log(f'wrote {len(df)} rows x {df.shape[1]} cols -> {OUTPUT}')
