import mt_cleaning_utils as u

INPUT = 'mt_data/mt_records_anonymized.csv'
OUTPUT = 'mt_data/mt_records_cleaned_complete_raw.csv'
COLUMNS_FILE = 'mt_columns.json'

KEY = 'provider_id'   # provider_number is renamed to provider_id in finalize
TARGET = 'qr_rating'  # star_level is renamed to qr_rating in finalize

# Discovered-at-runtime column families retained by finalize(). Same raw
# scaffold as mt_clean_raw.py: complete == raw preprocessing, only the target
# row filter differs (invalid/missing ratings are kept here).
DYNAMIC_PREFIXES = ('name_',)


if __name__ == "__main__":
    # Reads as str (IDs and zip codes must never be coerced) and hard-fails on
    # a stale records file.
    df = u.read_records(INPUT)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = u.strip_dollar_prefix(df[col])

    # program_name has no delimiter, so decompose it by keyterm match into one
    # TEXT column per term (the term where present, else NaN).
    df = u.adopt_keyterm_columns(df, 'name', as_bool=False)

    # The categorical columns (program_type, provider_type, county,
    # ccrr_region) and provider_id_source are already atomic, human-readable
    # text and are kept as-is via the stable scaffold in mt_columns.json.

    # valid_target_values is omitted, so finalize() keeps every row, including
    # the "Pre-Star" programs (accepted into STARS but not yet rated), which
    # survive here as the literal text "Pre-Star".
    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True)
    df.to_csv(OUTPUT, index=False)
