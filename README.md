# CCQ: A Multi-State Child Care Quality Dataset

CCQ is a de-identified, provider-level dataset of child care quality ratings (QR)
from the Quality Rating and Improvement Systems (QRIS) of 12 U.S. states. It
covers **59,372 providers**, of which **29,073 (49.0%)** carry a published
quality rating. This repository contains the code used to collect and curate the
dataset and to run the benchmark experiments reported in the paper.

## Dataset

| State | QRIS | QR scale | #Provider | #Rated | #Feature (raw) | #Feature (preprocessed) |
|---|---|:---:|---:|---:|---:|---:|
| CA | Quality Counts California | 1–5 | 10,715 | 847 | 142 | 84 |
| CO | Colorado Shines Rating | 1–5 | 4,508 | 3,423 | 64 | 119 |
| GA | Quality Rated | 1–3 | 8,192 | 2,897 | 203 | 280 |
| KY | Kentucky All STARS | 1–5 | 1,929 | 1,896 | 61 | 114 |
| MD | Maryland EXCELS | 1–5 | 7,106 | 5,005 | 32 | 36 |
| MT | Best Beginnings STARS to Quality | 1–5 | 213 | 181 | 12 | 22 |
| NC | NC Star Rated License | 1–5 | 3,264 | 3,050 | 30 | 32 |
| NE | Step Up to Quality | 1–5 | 3,274 | 1,079 | 38 | 89 |
| OK | Reaching for the Stars | 1–5 | 2,557 | 2,554 | 179 | 28 |
| SC | ABC Quality | 1–5 | 2,400 | 1,223 | 10 | 56 |
| WA | Early Achievers | 1–5 | 10,399 | 3,168 | 169 | 109 |
| WI | YoungStar | 1–5 | 4,815 | 3,750 | 164 | 38 |
| **Total** | | | **59,372** | **29,073** | | |

Feature counts include `provider_id` and `qr_rating`. Each state is released in
two formats, row-aligned within the state:

- **raw** (`{st}_records_cleaned_raw.csv`): minimal preprocessing; the source
  text is preserved so records can be serialized for language models.
- **preprocessed** (`{st}_records_cleaned_full.csv`): strictly numeric and
  boolean, ready for standard machine-learning pipelines.

Every file starts with `provider_id` (a surrogate id, read as a string) and
`qr_rating` (on the state's native scale; ratings are not comparable across
states).

The dataset is hosted on Hugging Face as a gated dataset under the CC-BY-NC-SA
4.0 license: <https://huggingface.co/datasets/jiayinglu/CCQ>.

## Repository structure

```
data-collection/          dataset construction (Section 3)
  README.md               pipeline overview and how to run it
  requirements.txt
  regenerate_release.py   rebuild the cleaned files for all states
  prompts/                prompts for collection, anonymization and cleaning
  {st}/                   one folder per state: crawler, anonymizer, cleaning
    README.md             seed source and how to run that state
experiments/              benchmark (Sections 4-5)
  README.md               setup and commands for every experiment
  requirements.txt
  data/                   the 24 released CSVs go here
  *.py
```

The two folders are independent; the only link between them is the set of
released CSVs. To reproduce the benchmark, download the dataset into
`experiments/data/` and follow [`experiments/README.md`](experiments/README.md).
To see how the data were collected and curated, start with
[`data-collection/README.md`](data-collection/README.md).
