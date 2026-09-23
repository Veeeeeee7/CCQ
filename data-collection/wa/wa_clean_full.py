import os
import pandas as pd
import wa_cleaning_utils as u

INPUT = 'wa_data/wa_records_anonymized.csv'
OUTPUT = 'wa_data/wa_records_cleaned_full.csv'
COLUMNS_FILE = 'wa_columns.json'
LOG_FILE = 'wa_cleaning_log_full.txt'

KEY = 'provider_id'   # Salesforce `Id` is renamed to provider_id in finalize
TARGET = 'qr_rating'  # Early_Achiever_Status_Internal__c -> qr_rating in finalize

# Level 3+ collapses to 3 and non-rating statuses become NaN (normalize_rating).
# There is no Level 1 — Level 1 is simply being licensed.
VALID_RATINGS = (2, 3, 4, 5)

# Discovered-at-runtime boolean families retained by finalize().
DYNAMIC_PREFIXES = ('agegroup_', 'slotavail_', 'spec_', 'language_', 'langinstr_',
                    'factype_', 'status_', 'licstatus_', 'lictype_', 'certtype_')


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

    # The target mixes levels with non-rating statuses; reduce it to a number.
    df = u.normalize_rating(df, log=log)

    df = df.rename(columns=u.FEATURE_RENAME)

    # Nested JSON -> numeric counts + severity features.
    df = u.json_counts(df)

    # A blank *_count means the detail page was never seen: unknown, not zero.
    df = u.mask_unknown_counts(df, log=log)

    # Structured text -> numeric/boolean scalars. Their text byproducts are not
    # in the full scaffold, so finalize() drops them.
    df = u.parse_ages_served(df, log=log)
    df = u.parse_contact_blob(df, log=log)
    df = u.parse_hours(df)
    df = u.parse_license_dates(df)
    df = u.presence_flags(df)

    # Single-value categoricals -> one-hot booleans over discovered values.
    df, _ = u.build_categorical_onehot(df, 'facility_type', 'factype')
    df, _ = u.build_categorical_onehot(df, 'provider_status', 'status')
    df, _ = u.build_categorical_onehot(df, 'license_status', 'licstatus')
    df, _ = u.build_categorical_onehot(df, 'license_type', 'lictype')
    df, _ = u.build_categorical_onehot(df, 'license_certificate_type', 'certtype')

    # Multi-value text -> presence booleans over discovered items.
    df, _ = u.build_multivalue_columns(
        df, 'age_groups_served', delimiter=';', prefix='agegroup', as_bool=True)
    df, _ = u.build_multivalue_columns(
        df, 'slot_availability', delimiter=';', prefix='slotavail', as_bool=True)
    df, _ = u.build_multivalue_columns(
        df, 'ea_specialization', delimiter=';', prefix='spec', as_bool=True)
    df, _ = u.build_multivalue_columns(
        df, 'languages_spoken', delimiter=';', prefix='language', as_bool=True)
    df, _ = u.build_multivalue_columns(
        df, 'languages_of_instruction', delimiter=';', prefix='langinstr',
        as_bool=True)

    # 'True'/'False' and 'Yes'/'No' strings -> real booleans.
    df = u.to_bool(df)

    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES,
                    valid_target_values=VALID_RATINGS)

    # Uses post-rename names, so it has to run after finalize().
    df = u.cast_int_columns(df, log=log)
    df.to_csv(OUTPUT, index=False)
    log(f'rows={len(df)} cols={len(df.columns)} -> {OUTPUT}')
