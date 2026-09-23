"""
ok_seed.py — build the Oklahoma provider seed from OKDHS's Child Care Locator.

The Child Care Locator (https://ccl.dhs.ok.gov/) is a Next.js app whose
/providers search page is server-rendered, so plain `requests` returns the
result cards with their license numbers.

The search always covers a fixed ~5 mile radius around the geocoded location:
the `radius` parameter is ignored server-side, and "<County> County, OK" often
geocodes to a point far from where providers cluster. Coverage therefore comes
from search-point density: every incorporated Oklahoma city/town is searched by
name ("<Place>, OK"). The 597 places are Wikipedia's "List of municipalities in
Oklahoma" (Census SUB-IP-EST2024-POP-40); all 77 counties are represented.

Only provider_id (the license number, e.g. "K830023010") is collected here;
everything else comes from the detail page in ok_crawler.py.

Usage:
    python ok_seed.py --places "Altus,Lawton,Enid" --limit 3   # smoke test
    python ok_seed.py                                          # full 597-place sweep
    python ok_seed.py --merge-only                              # rebuild ok_seed.csv from seed_raw/ cache
    python ok_seed.py --places Altus --force                    # re-pull one place

Deps: pip install requests pandas beautifulsoup4
"""

import argparse
import collections
import os
import random
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://ccl.dhs.ok.gov'
SEARCH_PATH = '/providers'

OUT_DIR = 'ok_data'
RAW_DIR = os.path.join(OUT_DIR, 'seed_raw')
MERGED_PATH = os.path.join(OUT_DIR, 'ok_seed.csv')
LOG_FILE = 'ok_seed_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# Every incorporated Oklahoma city/town (see module docstring).
OK_PLACES = [
    'Oklahoma City', 'Tulsa', 'Norman', 'Broken Arrow', 'Edmond', 'Lawton',
    'Moore', 'Midwest City', 'Enid', 'Stillwater', 'Owasso', 'Bartlesville',
    'Muskogee', 'Shawnee', 'Bixby', 'Jenks', 'Yukon', 'Ardmore', 'Ponca City',
    'Mustang', 'Sapulpa', 'Duncan', 'Del City', 'Durant', 'Claremore',
    'Bethany', 'Sand Springs', 'El Reno', 'Altus', 'McAlester', 'Tahlequah',
    'Chickasha', 'Ada', 'Newcastle', 'Glenpool', 'Miami', 'Guymon', 'Choctaw',
    'Weatherford', 'Woodward', 'Guthrie', 'Elk City', 'Okmulgee', 'Coweta',
    'Warr Acres', 'Blanchard', 'Collinsville', 'Pryor Creek', 'The Village',
    'Poteau', 'Piedmont', 'Skiatook', 'Sallisaw', 'Tuttle', 'Cushing',
    'Wagoner', 'Clinton', 'Noble', 'Catoosa', 'Grove', 'Seminole', 'Idabel',
    'Purcell', 'Harrah', 'Tecumseh', 'Pauls Valley', 'Blackwell',
    'Holdenville', 'Verdigris', 'Henryetta', 'Anadarko', 'Vinita',
    'Lone Grove', 'Kingfisher', 'Hugo', 'Alva', 'Hinton', 'Sulphur', 'Sayre',
    'Pocola', 'McLoud', 'Marlow', 'Perry', 'Slaughterville', 'Bristow',
    'Broken Bow', 'Madill', 'Roland', 'Spencer', 'Stilwell', 'Nichols Hills',
    'Fort Gibson', 'Elgin', 'Nowata', 'Goldsby', 'Dewey', 'Mannford',
    'Muldrow', 'Frederick', 'Hobart', 'Perkins', 'Hominy', 'Bethel Acres',
    'Cleveland', 'Jones', 'Cache', 'Calera', 'Checotah', 'Tishomingo',
    'Okemah', 'Heavener', 'Wewoka', 'Tonkawa', 'Chandler', 'Atoka',
    'Pawhuska', 'Lindsay', 'Marietta', 'Eufaula', 'Stroud', 'Davis',
    'Stigler', 'New Cordell', 'Mangum', 'Fairview', 'Watonga', 'Drumright',
    'Jay', 'Prague', 'Walters', 'Healdton', 'Nicoma Park', 'Wilburton',
    'Kiefer', 'Commerce', 'Newkirk', 'Hennessey', 'Spiro', 'Antlers', 'Pink',
    'Chouteau', 'Union City', 'Krebs', 'Lexington', 'Chelsea', 'Pawnee',
    'Wynnewood', 'Burns Flat', 'Inola', 'Hartshorne', 'Arkoma', 'Waurika',
    'Hooker', 'Haskell', 'Coalgate', 'Langston', 'Luther', 'Hollis',
    'Granite', 'Warner', 'Minco', 'Helena', 'Kingston', 'Cherokee', 'Wilson',
    'Stratford', 'Westville', 'Byng', 'Locust Grove', 'Comanche', 'Crescent',
    'Vian', 'Carnegie', 'Dickson', 'Oologah', 'Morris', 'Waukomis', 'Panama',
    'Beaver', 'Fletcher', 'Konawa', 'Sperry', 'Central High', 'Snyder',
    'Laverne', 'Okarche', 'Shattuck', 'Beggs', 'Geronimo', 'Thomas',
    'Mooreland', 'Wetumka', 'Fairland', 'Salina', 'Colbert', 'Fairfax',
    'Maysville', 'Boley', 'Boise City', 'Caddo', 'Yale', 'West Siloam Springs',
    'Wister', 'Okeene', 'Kellyville', 'Forest Park', 'Meeker', 'Rush Springs',
    'Buffalo', 'Apache', 'North Enid', 'Shady Point', 'Gore', 'Bray', 'Geary',
    'Dibble', 'Mounds', 'Erick', 'Barnsdall', 'Cashion', 'Talihina',
    'Goodwell', 'Oakland', 'Medford', 'Hydro', 'Pond Creek', 'Oilton',
    'Grandfield', 'Ringling', 'Maud', 'Davenport', 'Quinton', 'Temple',
    'Texhoma', 'Tipton', 'Quapaw', 'Ninnekah', 'Seiling', 'Weleetka',
    'Valliant', 'Cyril', 'Allen', 'Washington', 'Dewar', 'Sentinel',
    'Colcord', 'Kansas', 'Afton', 'Elmore City', 'Rock Island', 'Adair',
    'Cheyenne', 'Empire City', 'Mannsville', 'Tyrone', 'Garber', 'Copan',
    'Mountain View', 'Morrison', 'Wellston', 'Blair', 'Springer', 'Waynoka',
    'Haileyville', 'South Coffeyville', 'Wayne', 'Ryan', 'Sterling',
    'Arapaho', 'Cole', 'Valley Brook', 'Porter', 'Howe', 'Welch', 'Langley',
    'Roff', 'Savanna', 'Bokchito', 'Earlsboro', 'Porum', 'Kiowa',
    'Wright City', 'Paoli', 'Corn', 'Okay', 'Vici', 'Boswell', 'Velma',
    'Billings', 'Carney', 'Clayton', 'Winchester', 'Ramona', 'Red Oak',
    'Lahoma', 'Glencoe', 'Ketchum', 'Verden', 'Alex', 'Calumet', 'Fort Cobb',
    'Wyandotte', 'Fort Towson', 'Hulbert', 'Canute', 'Arnett',
    'Medicine Park', 'Covington', 'Johnson', 'Ravia', 'Hammon', 'Drummond',
    'Olustee', 'Westport', 'Amber', 'Canton', 'Forgan', 'Keota', 'Foyil',
    'Ochelata', 'Binger', 'Cedar Valley', 'Bernice', 'Schulter',
    'Stringtown', 'Achille', 'Tushka', 'Paden', 'Cement', 'Stonewall',
    'Depew', 'Dill City', 'Gage', 'Dover', 'Thackerville', 'Chattanooga',
    'Bokoshe', 'Silo', 'East Duke', 'Ringwood', 'Prue', 'Tryon', 'Leedey',
    'Wapanucka', 'Wynona', 'Whitefield', 'Asher', 'Custer City', 'McCurtain',
    'Carmen', 'Lone Wolf', 'Spavinaw', 'Ripley', 'Bowlegs', 'Fort Coffee',
    'Bridge Creek', 'Oktaha', 'Katie', 'Tribbey', 'Webbers Falls', 'Coyle',
    'Sportsmen Acres', 'Sawyer', 'Vera', 'Tupelo', 'Cameron', 'Shidler',
    'Dustin', 'Kaw City', 'Optima', 'Agra', 'Bennington', 'Fort Supply',
    'Fanshawe', 'Eldorado', 'Mountain Park', 'Wanette', 'Calvin', 'Wakita',
    'Avant', 'Fargo', 'Crowder', 'Randlett', 'North Miami', 'Jennings',
    'Lamont', 'Mill Creek', 'Eakly', 'Kinta', 'Haworth', 'Lenapah', 'Lehigh',
    'Rattan', 'Terral', 'Cleo Springs', 'Gracemont', 'Watts', 'Taloga',
    'Delaware', 'Indiahoma', 'Ralston', 'Braggs', 'Oaks', 'Talala', 'Keyes',
    'Gans', 'Goltry', 'Francis', 'Milburn', 'Roosevelt', 'Foster', 'Red Rock',
    'Kremlin', 'Bluejacket', 'Mead', 'Davidson', 'Cromwell', 'Mulhall',
    'Disney', 'Soper', 'Marshall', 'Alderson', 'Millerton', 'Butler', 'Byars',
    'Hardesty', 'Breckenridge', 'Jet', 'Meno', 'Caney', 'Pocasset', 'Stuart',
    'Dougherty', 'Ames', 'Fitzhugh', 'Nash', 'Marble City', 'Bessie',
    'Camargo', 'Marland', 'Longdale', 'Carter', 'Freedom', 'Pittsburg',
    'Garvin', 'Osage', 'Warwick', 'Arcadia', 'Big Cabin', 'Taft', 'Aline',
    'Castle', 'Manitou', 'Boynton', 'Cimarron City', 'Etowah', 'Braman',
    'Gotebo', 'Kenefic', 'Martha', 'Gene Autry', 'Liberty', 'LeFlore',
    'Tamaha', 'Woodlawn Park', 'Spaulding', 'Slick', 'Indianola', 'Canadian',
    'Hunter', 'Cowlington', 'Orlando', 'Bearden', 'Grayson', 'Kemp',
    'Reydon', 'Fairmont', 'Peoria', 'Sharon', 'Lawrence Creek', 'Phillips',
    'Rocky', 'St. Louis', 'Carlton Landing', 'Burbank', 'Norge', 'Lamar',
    'Sparks', 'Armstrong', 'Bromide', 'Burlington', 'Tatums', 'Willow',
    'Greenfield', 'Cornish', 'Faxon', 'Hallett', 'Summit', 'Tullahassee',
    'Council Hill', 'Dacoma', 'Colony', 'Lookeba', 'Hastings', 'Pensacola',
    'Rentiesville', 'Loco', 'Hanna', 'Smithville', 'Foss', 'Sweetwater',
    'Redbird', 'Bridgeport', 'Hitchcock', 'Wann', 'Wainwright', 'Gould',
    'Kendrick', 'Devol', 'Gerty', 'Horntown', 'Addington', 'Manchester',
    'New Alluwe', 'Carrier', 'Lake Aluma', 'Centrahoma', 'Kildare',
    'Hoffman', 'Atwood', 'Bradley', 'Terlton', 'Hickory', 'Grand Lake Towne',
    'Leon', 'Lone Chimney', 'Sasakwa', 'Deer Creek', 'Brooksville',
    'Fair Oaks', 'Headrick', 'Hillsdale', 'Blackburn', 'Loyal', 'Oakwood',
    'Paradise Hill', 'Maramec', 'Rosedale', 'Hendrix', 'Lima', 'Skedee',
    'Strang', 'Ratliff City', 'Elmer', 'Erin Springs', 'Hitchita', 'Albion',
    'Mutual', 'Webb City', 'IXL', 'Gate', 'Douglas', 'Smith Village',
    'Yeager', 'Rosston', 'Oak Grove', 'Vernon', 'New Woodville', 'Byron',
    'Texola', 'Amorita', 'Hollister', 'Clearview', 'Grainola', 'Moffett',
    'Ashland', 'May', 'Strong City', 'Putnam', 'Fallis', 'Friendship',
    'Macomb', 'Sugden', 'Valley Park', 'Stidham', 'Renfrow', 'Foraker',
    'Meridian', 'Loveland', 'Jefferson', 'Lambert', 'Knowles', 'Cooperton',
    'Lotsee', 'Capron', 'Hoot Owl', 'Mule Barn',
]
assert len(OK_PLACES) == 597, f'expected 597 OK places, got {len(OK_PLACES)}'
assert len(set(OK_PLACES)) == 597, 'duplicate place name in OK_PLACES'

# Only the "View Provider Details" link is read from a card; everything else
# comes from the detail page.
_DETAIL_HREF_RE = re.compile(r'/providers/([A-Za-z0-9]{5,20})["\'?]')
_COUNT_RE = re.compile(r'([\d,]+)\s+providers?\s+match\s+your\s+filters', re.I)


def create_log_file(path=LOG_FILE):
    with open(path, 'w') as f:
        f.write('')


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def _slug(place):
    return re.sub(r'[^a-z0-9]+', '_', place.lower()).strip('_')


def fetch_place(place, session=None, timeout=30):
    """GET the Locator's search results for '<place>, OK' (no radius param;
    it is ignored server-side). Returns raw HTML, or None on request failure
    so a single bad place never kills the sweep."""
    session = session or requests
    params = {'location': f'{place}, OK'}
    try:
        r = session.get(BASE_URL + SEARCH_PATH, params=params,
                        headers={'User-Agent': UA}, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log(f'  ! request failed for {place}: {e}')
        return None


def parse_provider_ids(html):
    """Every distinct provider_id (license number) linked from a
    'View Provider Details' card on this results page, in document order."""
    ids = []
    seen = set()
    for m in _DETAIL_HREF_RE.finditer(html):
        pid = m.group(1)
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


def parse_stated_count(html):
    """The page's own 'N providers match your filters' number, to cross-check
    the parsed card count. None if not found."""
    soup = BeautifulSoup(html, 'html.parser')
    m = _COUNT_RE.search(soup.get_text(' '))
    if not m:
        return None
    try:
        return int(m.group(1).replace(',', ''))
    except ValueError:
        return None


# Per-place raw cache, so --merge-only can re-parse without re-fetching.

def raw_path(place):
    return os.path.join(RAW_DIR, f'{_slug(place)}.html')


def sweep(places, force=False, delay_range=(1, 2.5)):
    os.makedirs(RAW_DIR, exist_ok=True)
    session = requests.Session()
    for i, place in enumerate(places):
        path = raw_path(place)
        if os.path.exists(path) and not force:
            log(f'[{i}] {place}: cached, skipping fetch')
            continue
        log(f'[{i}] {place}: fetching')
        html = fetch_place(place, session=session)
        if html is None:
            continue
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        ids = parse_provider_ids(html)
        stated = parse_stated_count(html)
        if stated is not None and stated != len(ids):
            log(f'  ! MISMATCH {place}: page says {stated} providers match '
                f'but only {len(ids)} card(s) parsed -- possible pagination '
                f'or lazy-load; re-inspect this place by hand '
                f'(seed_raw/{_slug(place)}.html).')
        else:
            log(f'  {place}: {len(ids)} provider(s)'
                + (' (confirmed against stated count)' if stated is not None else
                   ' (page did not state a total to cross-check against)'))
        if i < len(places) - 1:
            time.sleep(random.uniform(*delay_range))


def merge(places, out_csv=MERGED_PATH):
    hits = collections.defaultdict(list)
    missing = []
    for place in places:
        path = raw_path(place)
        if not os.path.exists(path):
            missing.append(place)
            continue
        with open(path, encoding='utf-8') as f:
            html = f.read()
        for pid in parse_provider_ids(html):
            hits[pid].append(place)

    if missing:
        log(f'{len(missing)} place file(s) never fetched, skipped in merge: '
            f'{", ".join(missing)}')
    if not hits:
        raise RuntimeError(
            'Parsed 0 providers across all cached places -- the page '
            'markup likely changed (the /providers/<id> link pattern this '
            'script depends on is gone). Inspect a cached file in seed_raw/.')

    rows = [{'provider_id': pid, 'n_place_hits': len(ps),
             'seed_places': ';'.join(ps)} for pid, ps in hits.items()]
    df = pd.DataFrame(rows).sort_values('provider_id').reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f'Seed built: {len(df)} unique providers -> {out_csv} '
        f'(median {int(df.n_place_hits.median())} place hits/provider)')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Build Oklahoma's provider seed CSV from the Child Care Locator.")
    ap.add_argument('--places', default=None,
                    help='Comma-separated place names (default: all 597).')
    ap.add_argument('--force', action='store_true',
                    help='Re-fetch places even if a cached raw file exists.')
    ap.add_argument('--merge-only', action='store_true',
                    help='Skip fetching; just rebuild the seed CSV from seed_raw/.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only sweep the first N places (smoke test).')
    ap.add_argument('--output', default=MERGED_PATH)
    ap.add_argument('--delay-min', type=float, default=1)
    ap.add_argument('--delay-max', type=float, default=2.5)
    args = ap.parse_args()

    create_log_file()
    places = ([p.strip() for p in args.places.split(',')] if args.places
              else list(OK_PLACES))
    for p in places:
        if p not in OK_PLACES:
            raise SystemExit(f'Unknown place: {p!r}')
    if args.limit:
        places = places[:args.limit]

    if not args.merge_only:
        sweep(places, force=args.force,
              delay_range=(args.delay_min, args.delay_max))
    merge(places, out_csv=args.output)
