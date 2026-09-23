import mt_cleaning_utils as u

INPUT = 'mt_data/mt_records_anonymized.csv'
OUTPUT = 'mt_data/mt_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'mt_columns.json'

KEY = 'provider_id'
TARGET = 'qr_rating'

DYNAMIC_PREFIXES = ('name_',)


if __name__ == "__main__":
    df = u.read_records(INPUT)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = u.strip_dollar_prefix(df[col])

    # name_* keyterm columns: the term where present, else NaN.
    df = u.adopt_keyterm_columns(df, 'name', as_bool=False)

    # No valid_target_values: every row is kept, "Pre-Star" as literal text.
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df.to_csv(OUTPUT, index=False)
