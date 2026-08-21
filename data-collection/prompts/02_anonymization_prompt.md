# Prompt 2 — Anonymization

You are given one U.S. state's collected provider table. Your job is to decide which of its columns must not survive and used in anything built from it, and to say why.

**This stage produces no code.** Your deliverable is a **list of columns to remove, each with a one-line reason**. Do not write a script to apply it, do not offer to, and do not modify any CSV. Deciding what a column means is the work; applying the decision is not.

---

## 0. Your input

| attachment         | role                                                                    |
| ------------------ | ----------------------------------------------------------------------- |
| `{st}_records.csv` | the state's collected table, in full — every row, every column          |
| `ga_records.csv`   | a finished Georgia crawl, for comparison                                |
| `ga_anonymize.py`  | Georgia's finished drop list, so you can see the shape yours ends up in |

**You have the whole table.** Open it and read it as a table. Every verdict you reach should be one you confirmed against real cells — "this looks like a phone number" is something to check, not something to assert.

**Checking a column is cheap, so guessing is not defensible.** Before calling a column private, look at it. Before calling one a rating restatement, cross-tabulate it against the rating column and see whether they move together. If you are reasoning from a column name, you skipped the step that was made available to you.

Georgia is attached so you can compare a table you are seeing for the first time against one whose treatment is already settled — the same kind of column often appears under a different name.

---

## 1. What you are looking for

Two classes, both **column-level**. Nothing here filters rows, so the anonymized file has exactly the rows of the input, in the same order.

| class                      | what goes                                                                                |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| **P — private**            | anything that identifies a person or a physical place rather than describing the program |
| **L — rating restatement** | anything that tells you the rating, or tells you that a rating happened                  |

Two columns are never removed, in any circumstance: the native column that becomes **`provider_id`** and the native column that becomes **`qr_rating`**.

---

## 2. Class P — private

The dataset is meant to describe programs, not the people who run them or the buildings they occupy. Remove a column when what it holds is a person, a place, or a means of contacting either.

**People and contact details.** Administrator, director, owner and contact names. Phone numbers, email addresses, personal, per-provider websites, or names.

**Places.** Street addresses, mailing addresses, and latitude/longitude. Remove coordinates outright — do not round them or swap in a centroid. A coordinate pair at this precision names a specific building, and rounding it does not stop that. It only makes the removal harder to verify.

Do not remove coarse geography. County, city, zip, region and school district describe where a program sits in the state's administrative structure rather than pointing at an address, they are ordinary public facts about a licensed facility, and they carry real information about the program. They stay and flow to cleaning as ordinary categorical features.

**Per-provider free text.** Inspection narratives, complaint and injury write-ups, monitoring notes, violation descriptions, adverse-action text — any column whose cells are prose written about one specific provider. These go for the same reason as a name column: prose written about a single facility names the people in it, quotes the address, and reads back as a story about an identifiable place. Trying to scrub the names out of it and keep the rest is not a thing you can verify, so the whole column goes.

Distinguish this from **categorical or coded text**, which stays: a violation type, a rule number, a program-status label, a list of accepted subsidies. The test is whether the cell is a sentence about one provider or a value drawn from a vocabulary the state reuses across providers.

**Per-provider URLs and keys.** A report URL or an internal record key that resolves to one provider is another copy of the identifier.

---

## 3. Class L — rating restatement

**The rule.** No column may survive that is the rating in different clothes, or that is a by-product of the rating being awarded. Only the one column that becomes `qr_rating` stays.

**The rating under another name.** A second column holding the same score, a star string, a rating band, a display label, a "current rating" field that duplicates the one you already picked as the target. If two columns move together and one of them is the target, the other one goes.

**The paperwork the rating generates.** Award dates, effective dates, expiration dates, renewal dates, rating status, rating level history, "date last rated", time until the rating expires, how old the rating is, and anything computed from those.

Be careful with this class, because it hides in the _absence_ of a value as much as in the value. If a state only issues a rating certificate to providers above a certain level, then whether an award date exists is itself a statement about the rating — and that survives every transformation you could apply to the column. Recoding it, bucketing it or replacing it with a "has an award date" flag all preserve exactly the thing that needs to go. Remove the column.

**What stays: the licensing lifecycle.** Licence issue dates, licence age, inspection dates, time since last inspection, inspection counts. Licensing is a separate regulatory process that runs whether or not a provider participates in the rating system, and it applies to rated and unrated providers alike. Keep these — and say in your report that you kept them and why.

---

## 4. How to decide

Read the column name for the hypothesis and the column itself for the answer. The name tells you where to look; only the data tells you what is there. `school_district_operated_program` sounds like a district identifier and is actually a yes/no about governance. A column called `notes` may hold three repeated status codes, or it may hold paragraphs. Look.

Take every column in turn and ask, in order:

1. **Is this about a person?** Names, contacts, anything addressed to a human.
2. **Is this about a place, precisely?** An address or a coordinate goes; a county does not.
3. **Is this cell a sentence about one provider, or a value from a shared vocabulary?** Sentences go. Distinct-count against row-count settles it: a value drawn from a vocabulary repeats, prose does not.
4. **Would knowing this tell me the rating?** Including "tell me that there is no rating." Cross-tabulate the candidate against the rating column, and check its fill pattern against the rating too — a column that is populated only for rated providers reports the rating by being present at all.
5. **Is this a second copy of the identifier?** One distinct value per row, and it survives a join back to the grain.

If a column survives all five, keep it.

**A column is not one thing.** A single `notes` or `comments` field can hold a status code on most rows and a paragraph naming a family on a handful. Scan the long tail before you rule: sort by value length and read the extremes. The rare rows are where the identifying content hides, and they are invisible to any summary that reports the common case.

---

## 5. Your deliverable

Every row of every table below must cite something you actually saw in the file — a real value, a count, a fill rate. You have the data; a reason that could have been written from the column name alone is not a reason.

### 5.1 Remove — class P

| column       | what you saw                                         | reason              |
| ------------ | ---------------------------------------------------- | ------------------- |
| `admin_name` | 2,841 distinct over 2,897 rows; e.g. "Maria Delgado" | identifies a person |
| …            |                                                      |                     |

### 5.2 Remove — class L

| column          | what you saw                                                                                         | reason                         |
| --------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------ |
| `rating_status` | `Rated` 2,102 / `Pending` 44 / `Not participating` 751; blank in exactly the 751 rows with no rating | states whether a rating exists |
| …               |                                                                                                      |                                |

### 5.3 Keep, but flagged

Every column a reasonable reviewer would question, with the reason you kept it. Licensing dates and coarse geography are the two that come up every time. **Silence about a kept column is worse than a wrong drop.**

### 5.4 Uncertain — needs the human

Anything you could not resolve, with the specific question and what would settle it. A column that happens to track the rating is not automatically a restatement of it — a well-run program may score well on many things at once. A column that exists _because_ the rating was issued always is.

This is the one distinction the data alone may not settle, since correlation and causation look identical in a crosstab. Say what you saw, say why it is ambiguous, and ask. Everything else in this stage you can and should resolve yourself — "uncertain" is for genuine ambiguity, not for columns you did not open.

### 5.5 The drop list, as plain text

The column names, one per line, ready to apply. Produce this **only after §5.1–§5.4 have been answered** — see §7.

---

## 6. Notes for whoever applies the list

Not your code to write, but state these in the report so they do not have to be rediscovered. `ga_anonymize.py` is attached as the model — it is short on purpose.

- **Column-only.** Assert the row count is unchanged before writing. Rows and row order are untouched.
- **Never drop** the ID or rating column, whatever the list says. Guard it explicitly.
- **Missing columns are fine.** Drop with `errors="ignore"` so the script keeps working when the crawler's schema shifts.
- **Log every drop** — column name, class, reason — so the file has an audit trail that does not depend on this conversation surviving.
- **Output** `{st}_data/{st}_records_anonymized.csv`.

---

## 7. Pause-and-ask protocol

**Present §5.1–§5.4 and wait for answers.** Do not produce the final list before then. Ask specifically about:

- **Borderline private columns** — is the provider's own name a trading name or the only usable label? Is a secondary licence number an identifier or a legitimate feature?
- **Borderline rating columns** — see §5.4.
- **Anything that touches the ID or the rating column.** Never guess here.

---

## 8. What to report when you finish

1. the drop list, grouped by class, each entry with the evidence behind it;
2. the flagged-but-kept columns from §5.3;
3. anything still unresolved from §5.4;
4. the native ID and rating column names, which stay exactly as they are.

Then stop.
