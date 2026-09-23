import pandas as pd
import sc_cleaning_utils as u

INPUT = 'sc_data/sc_records_anonymized.csv'
OUTPUT = 'sc_data/sc_records_cleaned_complete_full.csv'
COLUMNS_FILE = 'sc_columns.json'

KEY = 'provider_id'   # permit_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # abc_level is renamed to qr_rating in finalize

# Discovered-at-runtime boolean families retained by finalize().
DYNAMIC_PREFIXES = ('permittype_', 'facilitytype_', 'county_')


if __name__ == "__main__":
    # permit_number must not round-trip through a float, and the exempt
    # providers' blank permits must stay '' rather than NaN.
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False, low_memory=False)

    df = u.normalize_source_columns(df)
    df = u.synthesize_exempt_ids(df)

    # Reads the letter abc_level, so it must run before map_rating().
    df = u.recode_facility_type(df)

    df['abc_level'] = u.map_rating(df['abc_level'])

    df, _ = u.build_categorical_onehot(df, 'permit_type', 'permittype')
    df, _ = u.build_categorical_onehot(df, 'facility_type_code', 'facilitytype')
    df, _ = u.build_categorical_onehot(df, 'county', 'county')

    # No valid_target_values: pending and unrated providers (<NA>) are kept.
    df = u.finalize(df, COLUMNS_FILE, 'full', KEY, TARGET, DYNAMIC_PREFIXES)
    df.to_csv(OUTPUT, index=False)
