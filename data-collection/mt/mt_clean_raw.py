import os
import mt_cleaning_utils as u

INPUT = 'mt_data/mt_records_anonymized.csv'
OUTPUT = 'mt_data/mt_records_cleaned_raw.csv'
COLUMNS_FILE = 'mt_columns.json'
LOG_FILE = 'mt_cleaning_log_raw.txt'

KEY = 'provider_id'
TARGET = 'qr_rating'

# "Pre-Star" (accepted but not yet rated) is not a valid rating.
VALID_RATINGS = (1, 2, 3, 4, 5)

DYNAMIC_PREFIXES = ('name_',)


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a') as f:
        f.write(message + '\n')
    print(message)


if __name__ == "__main__":
    create_log_file()
    df = u.read_records(INPUT)

    for col in df.columns:
        if df[col].dtype == object and df[col].apply(
                lambda x: isinstance(x, str) and x.startswith('$')).any():
            df[col] = u.strip_dollar_prefix(df[col])

    # name_* keyterm columns: the term where present, else NaN.
    df = u.adopt_keyterm_columns(df, 'name', as_bool=False)

    log(f'read {len(df)} rows; star_level values: '
        f'{sorted(df["star_level"].dropna().unique())}')

    df = u.finalize(df, COLUMNS_FILE, 'raw', KEY, TARGET, DYNAMIC_PREFIXES,
                    na_as_level=True, valid_target_values=VALID_RATINGS)
    df.to_csv(OUTPUT, index=False)
    log(f'wrote {OUTPUT}: {df.shape[0]} rows x {df.shape[1]} cols')
