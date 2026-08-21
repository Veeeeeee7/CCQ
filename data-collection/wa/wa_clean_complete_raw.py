import pandas as pd
import wa_cleaning_utils as u

INPUT = 'wa_data/wa_records_anonymized.csv'
OUTPUT = 'wa_data/wa_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'wa_columns.json'

KEY = 'provider_id'   # Salesforce `Id` is renamed to provider_id in finalize
TARGET = 'qr_rating'  # Early_Achiever_Status_Internal__c -> qr_rating in finalize

# Discovered-at-runtime column families retained by finalize().
DYNAMIC_PREFIXES = ('agegroup_', 'slotavail_', 'spec_', 'language_', 'langinstr_',
                    'complaints_', 'inspections_', 'licensehist_', 'contacts_')


def strip_dollar_prefix(series):
    """Remove leading '$' and comma separators from currency strings; everything
    else is untouched. Kept for parity with the GA pipeline; no-op on WA today."""
    def _clean(x):
        if isinstance(x, str) and x.startswith('$'):
            return x.replace('$', '').replace(',', '')
        return x
    return series.map(_clean)


if __name__ == "__main__":
    df = pd.read_csv(INPUT, low_memory=False)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = strip_dollar_prefix(df[col])

    # The target mixes levels with non-rating statuses; reduce it to a number.
    # Unrated providers keep a NaN qr_rating and survive into this output.
    df = u.normalize_rating(df)

    # Feature columns are renamed inside finalize(), so the parsers below read
    # the native crawler names and write their tidy outputs.
    df = df.rename(columns=u.FEATURE_RENAME)

    # Structured text fields -> their decomposed parts, text preserved.
    df = u.parse_ages_served(df)
    df = u.parse_contact_blob(df)
    df = u.parse_location(df)
    df = u.parse_hours(df)
    df = u.parse_license_dates(df)

    # Multi-value text -> one text column per discovered item (the phrase where
    # present, else NaN). Schema is discovered from the data, not hardcoded.
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

    # Record counts accompany the decomposed text (the severity features are a
    # full-set concern and are left out of the raw scaffold).
    df = u.json_counts(df)

    # valid_target_values is omitted, so finalize() keeps every row, including
    # rows whose qr_rating is missing (Not Enrolled, Withdrawn, Rating Expired,
    # Participating-not-yet-rated, ...).
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df.to_csv(OUTPUT, index=False)
