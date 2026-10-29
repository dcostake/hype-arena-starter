# Kolmogorov team 56: a Hype Arena system

The system does the two jobs of the challenge:

1. **It labels rows.** Every row of the input whose `label` is empty gets `true`, `false` or `unsure`, and every `true` or `false` gets a confidence from 0.5 to 1.
2. **It writes new content.** Exactly 1,000 new claims about Hooly, each true, each in the forms of `contracts/claim-forms.md`.

It is not a language model. It reads sentences with deterministic rules, and three small log-linear models learned from the practice data resolve the facts and weigh the evidence. Everything runs on the Python standard library.

## Contents

- [Run it](#run-it)
- [The idea: a statement is its form plus its residue](#the-idea-a-statement-is-its-form-plus-its-residue)
- [One sentence, step by step](#one-sentence-step-by-step)
- [How well it does](#how-well-it-does)
- [The files](#the-files)
- [Every function, module by module](#every-function-module-by-module)

## Run it

| What | Command |
| --- | --- |
| The arena run: reads `/universe/`, writes `/output/output.csv` | `docker build --platform linux/amd64 -t kolmogorov56 .` then `scripts/check.sh kolmogorov56` |
| The frontend: upload a CSV, watch each step, download the tables | `python -m hype.server`, then open http://localhost:8000 |
| The arena run on this machine, without Docker | `python -m hype.run --contract universe.csv --out output.csv` |
| Label a table and get it back in the practice shape: `id, text, date, label, confidence, form, residue` | `python -m hype.run my_rows.csv --out results.csv` |
| Write sentences one by one; each is labelled and added to the table | `python -m hype.run --write` |
| Label given sentences and add them to the table | `python -m hype.run --add "2026-04-01 \| Hooly AI Pro costs €59 per month."` |
| Explain the label of a sentence in full | `python -m hype.pipeline "2026-04-01 \| Hooly AI Pro costs €59 per month."` |
| The 1,000 claims alone, with examples | `python -m hype.claims --show 5` |
| An Excel workbook of the system's labels against the practice labels | `python -m hype.report` (needs `openpyxl`) |
| Check the claims of a table against the schema and the lexicon | `python scripts/validate_claims.py output.csv` (needs `jsonschema`) |
| Measure each stage on held-out practice pages | `python -m hype.characterize --audit`, `python -m hype.resolve --eval`, `python -m hype.label --eval` |
| Learn the models again from the practice data | `python -m hype.resolve --train`, then `python -m hype.label --train` |

## The idea: a statement is its form plus its residue

The practice data gives each sentence a **form**: the facts it states, each a subject, an attribute and a value. A sentence says more than its form, though, and the rest is the **residue**:

> p-48 · *His group now counts 17 engineers, spread across Berlin and Lisbon.*
>
> **form:** `Holdan.team_size = 17 engineer` · **residue:** names `Berlin`, `Lisbon`; the reference `His`

The form is what the row is judged on. The residue is what the form leaves out but the system needs. Here `His` says the subject is a person named earlier on the page, and `Berlin` and `Lisbon` are cities the data has seen as headquarters and event cities.

A row is `false` when one of its facts did not hold on its date. It is `unsure` when the data gives no means to tell. Otherwise it is `true`. Each fact of the universe holds one value at a time and changes at dates. Hooly Gemini speaks 6 languages until Hooly I/O 2026, then 12. So a sentence is checked against the timeline of each of its facts, built from the labelled rows.

## One sentence, step by step

The sentence is new, dated 2026-07-18: *Hooly Pixel 10 retains its 4,970 mAh battery and delivers the same 31 hours of typical battery life advertised at launch.*

| Step | Module | What happens to the sentence |
| --- | --- | --- |
| 0. Pages | `resolve.py` | Consecutive rows of one date form one page. This sentence stands alone. |
| 1. Characterize | `characterize.py`, `values.py`, `lexicon.py` | Subject `Hooly Pixel 10`; values `4,970 mAh` (a quantity) and `31 hours` (a duration); reference `its`. |
| 2. Resolve | `resolve.py` | `4,970 mAh` becomes `battery_capacity`, `31 hours` becomes `battery_life`. **form:** both facts. **residue:** `its`. |
| 3. Knowledge | `knowledge.py` | `battery_capacity = 4,970 mAh`: a labelled-true row of the same day agrees, so it is *confirmed*. `battery_life = 31 h`: 31 h held at launch, but a labelled-true row of the same day says 34 h, so it is *stale*. |
| 4. Label | `label.py` | P(capacity holds) = 1.00, P(battery life holds) = 0.14, so P(row true) = 0.14. The row model gives true 0.04, false 0.96, unsure 0.00. A rule agrees: a labelled-true row of the same day contradicts a surely read fact. |
| Verdict | | **false, confidence 0.96** |

## How well it does

Measured on practice pages the models never saw, the way the arena run sees new rows: those pages carry no label and no form, and the system reads them itself.

| Measure | Result |
| --- | --- |
| Values of the gold forms the reader finds in the text | 99.9% |
| Facts resolved exactly (subject, attribute and value), F1 | 0.86 |
| Row labels right | **87.6%**; always `true` gets 83.3% |
| `false` rows: precision / recall | 0.72 / 0.48 |
| Calibration of the confidences (Brier score, lower is better) | 0.097 |
| Claims that more data overturns, with 20% of the pages held out | 0.0% to 0.6% |
| Claims valid against `entry.schema.json` and the lexicon | 1,000 of 1,000 |
| Arena run on 20,000 rows | under 3 minutes on a laptop |

Two rules hold without exception in the practice data, and the labeller applies them:

- a value contradicted by a labelled-true row of the same day is false (17 of 17);
- a value other than the one labelled-true rows hold on both sides of its date is false (81 of 81).

What it does not do yet: `unsure` is predicted only for a value the dataset itself marks unsure on the same day, and it does not check relations between two facts such as *twice as long as* or *ten monthly payments*.

## The files

| Path | What it is |
| --- | --- |
| `Dockerfile` | The image the arena runs: `python:3.14-slim`, the standard library only, `USER 10001`, `HOME=/tmp`, no `.pyc` written on the read-only root. |
| `hype/lexicon.py` | Names: the lexicon's subjects, aliases, attributes and events, and finding them in a text. |
| `hype/values.py` | Values: reading prices, counts, percentages, dates, versions and durations as `value-forms.md` reads them. |
| `hype/characterize.py` | Step 1: the profile of a sentence. Also the attribute index learned from the practice forms. |
| `hype/resolve.py` | Steps 0 and 2: pages, and from a profile to its facts. Also the log-linear model used everywhere. |
| `hype/knowledge.py` | Step 3: the observations of every fact and what they say about a value on a date. |
| `hype/label.py` | Step 4: the fact model, the row model, the rules, and their training and evaluation. |
| `hype/pipeline.py` | Steps 0 to 4 in one call, with the verdict of each row and its reasons; the form and the residue. |
| `hype/claims.py` | The 1,000 new claims: the evidence, the generator of each type, and the checker that reads each text back. |
| `hype/run.py` | The command line: the arena run, labelling a table, writing sentences. |
| `hype/server.py`, `hype/web/index.html` | The frontend. |
| `hype/report.py` | An Excel report of the system's labels against the practice labels. |
| `hype/model/resolver.json`, `hype/model/labeller.json` | The learned weights. |
| `scripts/validate_claims.py` | Checks the claims of a table against the schema and the lexicon. |

## Every function, module by module

Functions whose name starts with `_` are internal helpers of their module.

### `hype/lexicon.py`: names

| Function | What it does |
| --- | --- |
| `normalize_name(name)` | Lower case and no spaces, so `Hooly AI Pro` and `hoolyaipro` share one key. |
| `name_pattern(surface)` | A regular expression for a name as whole words, in any case, with the spaces between words optional. It does not match a name inside a version: `Hooly Android 16` is not found in `Hooly Android 16.0`. |
| `Subject` | One subject of the lexicon: its name, aliases and attributes, and whether it is an event. `surfaces` gives the name and all aliases. |
| `NameMatch` | One place in a text that names something: which subject, the spelling that matched, and where it stands. |
| `Lexicon.load(path)` | Reads `lexicon.json` and builds the patterns of every name and alias. |
| `Lexicon._index()` | Builds those patterns, longest name first. |
| `Lexicon.resolve(name)` | The subject a name or alias names, in any case and spacing, or `None`. |
| `Lexicon.has_attribute(subject, attribute)` | Whether the subject has this attribute in the lexicon. |
| `Lexicon.find(text)` | Every subject the text names. The longest name wins, so `Hooly AI Pro` names the plan and not the company Hooly. |
| `find_names(text, patterns)` | The matcher behind `find`: longest first, no two matches overlapping. Also used for names that are not subjects. |

### `hype/values.py`: values

| Function | What it does |
| --- | --- |
| `ValueMention` | One value the text writes: its kind and value (as `claimed_value.schema.json` writes them), where it stands, and how it is qualified (`op`, `negated`, `approximate`, `in_range`). `stated` says whether the referee reads it at all; `exact` says whether it is stated with no comparison. |
| `ValueReader(counted_nouns)` | The reader. `counted_nouns` are the nouns the data counts (`language`, `seat`, …), used to pick the noun after a number. |
| `ValueReader.read(text, masked)` | Every value in the text outside the masked spans (the names). It reads dates, then prices, percentages, versions, durations and quantities; each span is read once. |
| `ValueReader._dates()` | `2026-03-12`, `12 March 2026`, `March 12, 2026`, `March 2026`, a plain year; not `Q4 2026`. |
| `ValueReader._date_value(shape, match)` | Turns one matched date into `YYYY`, `YYYY-MM` or `YYYY-MM-DD`. |
| `ValueReader._money()` | Prices: `€`, `$`, `£`, `US$`, ISO codes, *euro*, *dollar* and *lei*; scales (`€12M`, `€1.2bn`); a decimal comma in a price; the period after the price (`/month`, `per year`, `monthly`, also after `per user`). |
| `ValueReader._period_before(text, start)` | A lenient reading of *its monthly price rose to €59* as €59 per month. Such a value is flagged `period_inferred`, because the referee reads a period only after the price. |
| `ValueReader._percents()` | `99.9%`, `50 percent`, `fifty per cent`. |
| `ValueReader._versions()` | `v2.3`, `version 2.3`, `2.3.1`, and a number with one point right after a capitalised name (`Hooly Gemini 2.2`), but not `4.1 stars` or `1.6 GB`. |
| `ValueReader._durations()` | `14 days`, `a 14-day trial`, `two weeks`, `24h`, and *a* as one only where it means one (*under a second*, not *1,000 requests a day*). |
| `ValueReader._duration(n, unit, start, end)` | Builds one duration, turning a fortnight into 2 weeks and a decade into 10 years. |
| `ValueReader._quantities()` | Physical units (`128 GB`, `5,000 mAh`, `6.1"`) and counted nouns (`40 languages`, `1,200 paying customers`). A plain four-digit year is never read as a count. |
| `ValueReader._counted_noun(rest)` | Which word after a number is its noun: up to two words may stand between them, and a noun the data counts wins (`4 data centre sites` counts sites). |
| `ValueReader._physical_unit(raw)` | A unit symbol or its spelling, keeping case where it matters: `m` is a metre and `M` a million. |
| `ValueReader._unit_follows(text, end)` | Whether a unit follows a number, so `4.1 stars` stays a quantity and not a version. |
| `ValueReader._qualify(v)` | Sets each value's comparison, negation, approximation, range and invisible-character flags. |
| `ValueReader._free`, `_claim`, `_mention` | Bookkeeping so that a span is read by one reader only. |
| `parse_number(s)` | `10,000` → 10000, `twelve` → 12, `a dozen` → 12. Returns `None` for a number that reads two ways, such as `1.500`. |
| `apply_scale(n, scale)` | Applies *thousand*, *million*, `M`, *billion*, `bn`, `k`. |
| `clean_number(n)` | Whole numbers as integers, without float noise such as 1,200,000.0000000002. |
| `singular(word)` | `languages` → `language`, `companies` → `company`, `people` → `person`. |
| `same_value(a, b)` | Whether two values are one after units convert: 14 days is 2 weeks, 1 GB is 1,000 MB, and a monthly price is not a one-off price. |
| `_number_words()` | The table of number words: *zero* to *ninety-nine*, *a dozen*, *a hundred*, *three hundred*. |
| `_negated(text, pos)` | Whether a negation (*not*, *no*, *never*, *without*, a word ending in *n't*) stands earlier in the value's clause. *No more than* is a comparison, not a negation. |
| `_op_before`, `_op_after` | The comparison words around a value: *over* is `gt`, *at least* is `ge`, *up to* is `le`, *under* is `lt`, *or more* after it is `ge`, and *before* or *after* a date. |
| `_approximate(text, start, end)` | *About*, *around*, *roughly*, `~` before the value, or *or so* and *-ish* after it. |
| `_in_range(text, start, end)` | One end of a range: `10-20 GB`, `between 10 and 20 GB`, `1 or 2 days`. |
| `_clause_start`, `_words_before`, `_ends_with` | Find the start of a clause and the words before a value. |

### `hype/characterize.py`: step 1, the profile of a sentence

| Function | What it does |
| --- | --- |
| `characterize_statement(row)` | The entry point: the `StatementProfile` of one row (`id`, `text`, `date`, and `form` if it has one). |
| `StatementCharacterizer.characterize(id, text, date, form)` | Builds the profile: layout, subjects, entities, values, references, time words, hedges, negations and attribute candidates. With a form, it also aligns each fact with the text and keeps the residue. |
| `StatementCharacterizer._layout(text)` | Whether the row is a sentence, a headline, a `Label: value` line or a quote, and who speaks (`Gavin: …`). |
| `StatementCharacterizer._is_person(name)` | Whether a name is a person: a subject with a `joined` date, or a name the data has seen as a CEO, CTO, lead or host. |
| `StatementCharacterizer._subjects(text)` | The subjects the text names. A lower-case one-word alias right after a number is not counted: *4,900,000,000 photos* counts photos, it does not name Hooly Photos. |
| `StatementCharacterizer._entities(text, …)` | Names that are not subjects. A known name comes with the facts it has been the value of (`Berlin` has been a headquarters and a city); otherwise it is a capitalised name not at the start of the sentence. |
| `StatementCharacterizer._references(text, …)` | What the sentence leaves to its page: *he*, *his*, *it*, *its*, *they*, *our*, *the editor*. Each reference says whether it points to a person, a thing or the speaker. |
| `StatementCharacterizer._temporal(text, …)` | Time words (*now*, *since*, *used to*, *will*), and months written without a year. |
| `StatementCharacterizer._epistemic(text, speaker)` | Who says it and how sure: *says*, *reportedly*, *might*, quotes, a speaker. |
| `StatementCharacterizer._attribute_candidates(…)` | The likeliest (subject, attribute) pairs, from the units, the names and the words of the sentence. |
| `StatementCharacterizer._align(profile, form)` | With a known form: finds each fact's value in the text, says whether its subject is named or comes from context, and keeps the residue. |
| `AttributeIndex` | What the lexicon and the practice forms say of each fact: which kinds and units its values take, which names it has had, and which words speak of it. Each fact's record is a `FactSignature`. |
| `AttributeIndex.from_rows(lexicon, rows)` / `build(lexicon, path)` | Learns the index from these rows only (so held-out rows stay unseen), or from a CSV. |
| `AttributeIndex.learn(text, form)` | Adds one row: its facts' kinds, units and names, and the words that stand with each attribute. |
| `AttributeIndex.compatible(value)` | The facts a value could be the value of, by its kind and unit: `17 engineers` fits `Holdan.team_size`. |
| `AttributeIndex.unseen_facts(kind)` | Facts of the lexicon that the practice data never shows, with the kind their attribute's name suggests. |
| `AttributeIndex.word_weight(word, attribute)` | A smoothed log-odds of how much a word speaks of an attribute (*refund* speaks of `refund_window`). |
| `AttributeIndex.kinds_of`, `facts_of`, `counted_nouns`, `name_patterns` | The kinds a fact takes, a subject's facts, the nouns the data counts, and patterns for the names it uses as values. |
| `guess_kind(attribute)` | The kind an unseen attribute's name suggests: `price` is money, `launched` a date, `is_…` yes or no. |
| `content_words(text, masked)` | The words that carry meaning, in the singular, without stop words and without the masked names. |
| `StatementProfile` and its parts | The profile itself. Every part is a `Span` of the text (its words, start and end): `SubjectMention`, `EntityMention`, `Reference`, `Marker`, `AttributeCandidate`, and with a form, `FactAlignment` and the residue. `to_dict()` gives it as JSON. |
| `default_characterizer()` | A characterizer on the practice lexicon and index, built once. |
| `_proper_runs(text)` | Runs of capitalised words, broken at months and common capitals; `Series A` stays one name. |
| `_negated_at`, `_inside`, `_dedupe` | Helpers: a negation before a position, overlap of spans, removing nested markers. |
| `_audit(rows, …)`, `main(argv)` | `--audit` measures how well the profiles explain the practice forms; `main` prints profiles. |

### `hype/resolve.py`: steps 0 and 2, from a profile to its facts

| Function | What it does |
| --- | --- |
| `segment_pages(rows)` | Step 0: runs of consecutive rows with one date, which are the pages of the universe. |
| `StatementResolver.resolve_page(page)` | Step 2 for one page: characterizes each row, then resolves its facts in two passes. The first pass reads each row with what came before it; the second also knows what the page's other rows seem to be about. |
| `StatementResolver.resolve(profile, context, models)` | One row: a *slot* for each value, name and yes-or-no attribute, each with the chance that it states no fact and its best candidates. |
| `StatementResolver._decode(slots)` | Chooses the facts: each slot states one fact at most, each fact takes one value, the surest first. |
| `StatementResolver._affinities(page)` | For each row, how much the page's *other* rows speak of each subject. This is how a page that never names Hooly Translate is still read as being about it. |
| `StatementResolver.fit(pages)` / `_fit_pass(…)` | Learns the three models of each pass from the practice forms: values, names, yes or no. |
| `StatementResolver.save(path)` / `load(lexicon, index, path)` | Writes or reads `hype/model/resolver.json`. |
| `StatementResolver._run`, `_fact`, `_name_mentions` | Runs one pass over a page; builds one resolved fact; lists the names that could be values. |
| `PageContext` | What the page has said so far: which subjects were named and how recently, the page's topic, the last person named, and the second pass's `affinity` for the current row. `advance()` moves to the next row; `recency()` gives 1 for the row before, 1/2 for the one before that, and so on. |
| `ResolvedFact` | One fact read from a sentence: subject, attribute, value, confidence, which words state it, and whether the subject was named or came from context. `as_form()` writes it as the practice forms do. |
| `FeatureBuilder.value_candidates(view, context, i)` | For value *i*: every compatible (subject, attribute), plus "no fact", each with its features. |
| `FeatureBuilder.name_candidates(…)`, `boolean_candidates(…)` | The same for a name that may be a value, such as Holdan as `reliability_lead`, and for yes-or-no attributes. |
| `FeatureBuilder.subject_features(…)` | Evidence for a subject: named in the sentence and how close to the value, recently on the page, the page's topic, part of its name written (*Berlin* for Hooly Berlin Office), a person for *his*. |
| `FeatureBuilder.attribute_features(…)` | Evidence for an attribute: the words of the sentence and those right around the value, the attribute's own words, the share of the fact's values in this unit or billing period. |
| `FeatureBuilder._null_features(view, v)` | Evidence that a value states no fact: negated, approximate, one end of a range, compared, no subject anywhere. |
| `FeatureBuilder._name_tokens`, `_unit_share`, `is_person`, `_attr_tokens` | Helpers for partial names, unit shares, persons, and an attribute's own words. |
| `RowView` | One row's words, label, references and mentions, worked out once for all its candidates. `near_words()` gives the words within a few words of a value. |
| `LogLinear` | The model used throughout: a score is a weighted sum of features. `probabilities()` is a softmax over candidates; `fit_ranking()` learns to rank the right candidate first (a conditional logit, with AdaGrad); `fit_binary()` and `binary()` do the same for yes or no. |
| `fact_matches(pred, gold, lexicon)`, `evaluate(resolver, pages)` | Whether a resolved fact is the gold one; precision, recall and F1 on pages. |
| `read_rows(path)`, `split_pages(pages)`, `main(argv)` | Read a CSV; split pages 80/20; `--eval`, `--train`, or resolve rows by id. |

### `hype/knowledge.py`: step 3, what was known and when

| Function | What it does |
| --- | --- |
| `KnowledgeBase.add_row(row, page, date, facts, label)` | Adds a row's facts as observations. A `true` row supports its values. A `false` row is a weak refutation of each of its values, since one of them is false. An `unsure` row marks its values as ones the data cannot check. An unlabelled row is weighted by the resolver's confidence. |
| `KnowledgeBase.evidence(fact, date, value, row, page)` | What the other rows say of this value of this fact on this date, never counting the row itself. It looks at the nearest support before and after the date and how far away each is, what other pages say on the same day, the weighted support on each side, refutations nearby, and whether the value is ever stated. |
| `KnowledgeBase._settle(…)` | What labelled-true rows of other pages fix: the value on the same day, and the value held on both sides of the date. These are the two rules. |
| `KnowledgeBase._agreement(…)` | The share of the supports on one day that agree with the value. |
| `KnowledgeBase.add`, `values_of`, `_sorted` | Add one observation; list a fact's observations by day. |
| `ClaimEvidence` | The answer of `evidence()`. `features()` turns it into features, including the combinations a linear model cannot form from its parts: *confirmed*, *stale* (held before, not any more), *change announced* (a new value later rows confirm), *contradicted alone*, *first then confirmed*. `settled_false` says whether a rule leaves no room for the value. |
| `Observation` | One observation: day, value, weight, whether it refutes, whether it is labelled or unsure, its row and page. |
| `values_agree(held, claimed)` | Whether a held value makes a claimed one true, comparisons included: `over 10,000` holds for 12,000; `2026-03` holds for 24 March 2026. |
| `names_match(a, b)` | `Gavin` and `Gavin Bellson` are one name once one holds the other's words. |
| `day_number(iso)` | A date as a day count; a year or a month counts as its first day. |
| `_magnitude(value)` | One number for an amount, across units, so values compare: bytes, seconds, days. |
| `value_key(value)`, `_plain(value)` | A key shared by the spellings of one value, for counting distinct values; a value without its comparison. |

### `hype/label.py`: step 4, the label

| Function | What it does |
| --- | --- |
| `Labeller.predict_all(evidence)` | The label, confidence and probabilities of each row. The row model decides first. Then the rules: a fact read surely (0.99 or more) that labelled rows leave no room for makes the row `false`, with at least 0.87, which is the rule's measured precision. A value only a same-day `unsure` row states makes it `unsure`. |
| `Labeller.fit(evidence, labels)` | Learns the fact model, then the row model, from rows whose labels are known. |
| `Labeller._settled_false(ev)`, `_known_unsure(ev)` | The two rules above. |
| `Labeller.save`, `load`, `_candidates` | Write or read `hype/model/labeller.json`; one feature set per label. |
| `FactModel` | `p_fact()`: P(fact holds) = sigmoid(weights · features). `p_row()` is the product over a row's facts: a row is true when all its facts hold. `fit()` learns from row labels as a noisy-OR, since we know which rows are false but not which of their facts. |
| `fact_features(kb, row)` | Each fact's evidence features, with the resolver's confidence and the value's kind. |
| `context_features(row)` | The row's own voice: layout, speaker, hedges, time words, negation, first person, exclamation. |
| `row_features(ev, facts_model, page)` | What the row model reads: P(row true) from the fact model, the weakest fact, which readings appear, the voice, and the page. |
| `page_scores(evidence, facts_model)` | For each row, the mean over its page's other rows: how false they look, how many stand alone, in which voice. A page that lies tends to lie throughout. |
| `collect_evidence(kb, rows)` | The features of every row, ready for the models. |
| `RowFacts`, `RowEvidence` | A row with its facts as read, and the same row turned into features. |
| `gold_facts(lexicon, row)` | The facts of a practice row's gold form. |
| `read_pages(resolver, pages, labelled, …)` | Pages as the run sees them: labelled pages with their gold forms, the others as the resolver reads them. |
| `build_knowledge(rows)` | A `KnowledgeBase` of these rows. |
| `out_of_fold_evidence(lexicon, pages, folds)` | Five folds of pages. Each fold is read by a resolver that never saw it, against the labelled rows of the other folds, as the arena run reads new rows. |
| `report(pred, gold)`, `main(argv)` | Accuracy, per-label precision and recall, calibration and confusion; `--eval`, `--train`. |

### `hype/pipeline.py`: steps 0 to 4 in one call

| Function | What it does |
| --- | --- |
| `HypePipeline.load(lexicon_path)` | Loads the lexicon, the two learned models and the 1,897 practice rows as labelled knowledge, with their gold forms. |
| `HypePipeline.analyse(rows, same_page, progress)` | The whole system on a table. Rows with a label are knowledge; rows without one get a verdict. It returns the verdicts and, for every row, its `Reading` (form and residue). `progress` hears each step as it ends, which is how the frontend shows the steps. |
| `HypePipeline.label(rows)` | The verdicts alone. |
| `HypePipeline.is_practice(row)` | Whether a row is one of the practice rows already known, with its gold form. |
| `HypePipeline._verdict(…)`, `_neighbours(…)` | Builds a verdict with its reasons, and the nearest labelled values before and after the date, for a person to check. |
| `residue(profile, facts)` | What the sentence writes that its form does not explain: values no fact takes, names that are neither a fact's subject nor its value, subjects named but unused, and references. |
| `Verdict` | Label, confidence, probabilities, who decided (`model`, `rule: settled false`, `rule: known unsure`), the facts with their reasons, and the values the referee would not read. `explain()` writes it out; `form()` gives the facts as a form. |
| `FactVerdict`, `Reading`, `READINGS` | One fact's verdict (how sure it was read, how likely it held, its reasons, its neighbours); a row's form and residue; the readings named to a person. |
| `format_value(value)`, `parse_line(line, n)`, `main(argv)` | Write a value for people; parse `DATE \| sentence`; the command line. |

### `hype/claims.py`: the 1,000 new claims

| Function | What it does |
| --- | --- |
| `generate_claims(lexicon_path, rows)` | The entry point: the evidence from the labelled rows, the generator, and the 1,000 claims. |
| `ClaimGenerator.generate(target)` | Builds every candidate of the five types, sorts each type by confidence, takes each type's share (compare 28%, change 24%, fact 24%, at_event 18%, relation 6%), then fills up with the surest. No fact appears in more than six compares. |
| `ClaimGenerator.facts(current)` | `fact`: a fact's value on 29 Oct 2026 (*The price of Hooly AI Pro is €59 per month.*). |
| `ClaimGenerator.compares(current)` | `compare`: two current values of one kind, units that convert, one currency and period. The facts must be related: one attribute of two subjects, two attributes of one subject that share a word, or two dates (*Hooly Workspace Enterprise costs more than Hooly AI Pro.*). |
| `ClaimGenerator._compare_texts(…)` | A few phrasings for a compare, the most natural first. The first one the checker accepts is kept. |
| `ClaimGenerator.changes()` | `change`: a value labelled true, then a different one labelled true later (*… went from version 1.4 to version 2.0.*). |
| `ClaimGenerator.at_events()` | `at_event`: a fact's value on an event's date, stated by labelled rows of that day or by agreeing labelled rows on both sides of it. |
| `ClaimGenerator.relations()` | `relation`: a fact of the entity a link names, such as *The CTO of Hooly joined on 1 July 2026*. The target's value must be the only one ever labelled, so it held while the link named the target. |
| `ClaimGenerator._keep(claim)` | Keeps a claim only if the checker finds no problem with its text, and counts why the others were dropped. |
| `ClaimGenerator.current_values()` | Every fact's value on 29 Oct 2026, with how sure it is. |
| `Evidence.current(fact, as_of)` | A fact's value on a date: its last labelled-true value, unless two labelled values disagree on that day, a labelled-false row later denies it, or the date is an event still to come. Sureness grows with the run of agreeing rows; an old count or price is less sure, a past launch date more. |
| `Evidence.held_on(fact, day)` | The value a fact held on a day: from labelled rows of that day, or from agreeing labelled rows on both sides. |
| `Evidence.add(row, facts)`, `sorted(fact)` | Add a labelled row's facts; a fact's observations by day. |
| `ClaimChecker.problems(type, text, form)` | Reads a claim's text back with the system's own reader and lists what the referee would refuse. It checks length, one sentence, not a copy of an input row, the subject named, the value read exactly and no other of its kind, no negation, no date or time word of its own, and for a compare no value or number at all, with the left subject first. |
| `value_text(value)` | A value written as `value-forms.md` reads it: `€59 per month`, `12 languages`, `14 days`, `12 March 2026`, `version 2.3`. Returns `None` when there is no safe written form. |
| `number_text(n)`, `plural(noun, n)`, `attribute_words(attribute)` | `2,048` with its comma (never a bare year); `company` → `companies`; `refund_window` → *refund window*, `cto` → *CTO*. |
| `fact_text(subject, attribute, value)` | The sentence of a fact: a verb for dates (*launched on*, *was released on*), *has* for counts of the attribute's own noun, otherwise *The … of … is …*. |
| `_related(fa, fb, kind)`, `_compare_group(value)` | Which two facts are worth comparing, and which values compare at all. Names and yes-or-no values never do. |
| `claim_rows(claims)` | The claims as rows of the contract: `id`, `type`, `text`, `form`, `label`, `confidence`. |
| `evidence_from(lexicon, rows, practice)` | The labelled rows of the input, with gold forms for the practice rows. Every input text is kept, so that no claim copies one. |
| `Claim`, `Observation`, `normalize_text`, `_outside`, `outside_subject`, `_plain`, `main` | The claim record, one labelled observation, text normalisation for the copy check, masking names, and the command line. |

### `hype/run.py`: the command line and the arena run

| Function | What it does |
| --- | --- |
| `run_arena(input, lexicon, output)` | The contract's run, in `/output/output.csv`: first a label row for every unlabelled input row, in the input's order (`confidence` 0.5 to 1 for `true` and `false`, empty for `unsure`, as the contract asks), then the 1,000 claims, each with its confidence. The table is written whole three times: every row `unsure` at once, then with the claims, then with the labels. A run stopped at the deadline still hands in a valid table. Its inner `label_rows()` writes the label rows of the contract. |
| `run_table(args)` | Labels a table and writes it back in the practice shape. When every row already has a label, it labels them all and compares. |
| `add_sentences(sentences, table)` | Labels each (date, sentence) and adds it to the end of a table, creating the table if needed. Usable from Python. |
| `run_add(args)`, `run_write(args)` | `--add` sentences on the command line; `--write` reads them from the keyboard, with `why` to explain the last one. |
| `practice_rows(rows, verdicts, readings)` | Rows in the practice shape: the system's label where there was none, and the form and residue everywhere. |
| `label_table(pipeline, rows, relabel)` | Runs the pipeline on a table, optionally ignoring the labels it has. |
| `read_table(path)`, `write_table(path, columns, rows)` | Read a CSV with `,` or `;`. Write one under another name and rename it when complete, so it is never half written. |
| `normalize_date(text)`, `readable(rows)` | Bring a date to `YYYY-MM-DD` from the ways tables write one (`2026/10/29`, and day first as Excel writes it in Romania: `29.10.2026`, `29/10/2026`); the rows with a text and a real date. A row whose date is no date is labelled `unsure`. |
| `label_safely(pipeline, rows)` | Labels the whole table; if that fails, page by page, so one page that breaks the system leaves only its own rows `unsure`. |
| `print_verdict`, `parse_sentence`, `table_columns`, `as_json`, `utf8_streams`, `main` | Print a verdict; parse `DATE \| sentence`; a table's columns; compact JSON; UTF-8 in and out whatever the console; the command line. |

### `hype/server.py` and `hype/web/index.html`: the frontend

| Function | What it does |
| --- | --- |
| `Handler.do_POST` / `Handler._run(rows, with_claims)` | `POST /api/run`: runs the pipeline on the uploaded CSV and streams one JSON line per step as it ends (pages, read, knowledge, labelled, claims, written), then the result. The result holds a preview of every row with its reasons, and the three tables as CSV. |
| `Handler.do_GET` | Serves the page, `/api/sample` (`sample/sample.csv` when present) and `/api/health`. |
| `parse_csv(text)`, `to_csv(columns, rows)` | Read an uploaded CSV and check its columns and dates; write rows as CSV text. |
| `Handler._send`, `_json`, `_emit`, `log_message`, `main` | Responses, streamed lines, quiet logging, and starting the server. |
| `index.html` | One page in the style of hype-arena.vercel.app. You upload the rows and see the system's steps light up as they run; that list is the architecture diagram. Then you see the result: totals, a table you can filter, and a click on a row for its facts, reasons, form and residue. There are downloads for the labelled table, `output.csv` and the claims. |

### `hype/report.py` and `scripts/validate_claims.py`: checking the work

| Function | What it does |
| --- | --- |
| `build(out, table)` | `report.xlsx`: each practice row's label from the system, once with all knowledge and once on unseen pages, beside the practice label. Differing rows are shaded, and a summary sheet holds accuracy, precision, recall and confusion. |
| `unseen_labels(lexicon, rows)` | Labels each practice page with a system that never saw it, which gives the honest number. |
| `results_to_excel(csv, out)` | Turns a table the system wrote into Excel, with the practice label beside the system's and the differences shaded. |
| `scores(pairs)`, `main` | Accuracy, precision, recall and confusion; the command line. |
| `scripts/validate_claims.py` | Checks every claim of a table against `entry.schema.json` and the lexicon (subject, attribute, link, target, event): what `check_output.py` leaves to the referee. |

---

# Hype Arena starter

## [hype-arena.vercel.app](https://hype-arena.vercel.app/)

## The universe

**Hooly** is a fictional company in a fictional universe, made for the QA DNA challenge of the NEXXT AI Hackathon. Its name alludes to Hooli, the tech giant of the TV series *Silicon Valley*. Its products carry the names of real Google products, used fictitiously: nothing here describes those products or their makers. Its people carry the names of the series' characters with a few letters changed.

Hooly's people, events, decisions and figures are invented, and the universe departs from the series and from Google's real history on purpose. **The truth is what the universe says.** A system that fills gaps from the web, or from what a pretrained model already knows, learns facts the universe contradicts.

The universe is a year of Hooly's pages, docs, posts, decks and press releases, in which prices, headcounts, dates and names change, and some pages lie. You get a part of it as sentences, each labelled.

## The task

Build a system that does two things with rows about Hooly, each with how sure your system is:

- **It labels rows.** Given rows of the universe, some labelled and some not, it labels each unlabelled row `true`, `false` or `unsure`.
- **It writes new content.** It writes exactly 1,000 new claims about Hooly that connect what the universe says, every one of them true.

How you build it is yours to decide.

- **LLMs may help you build your system,** and make what it learns from, but the system itself may not be an LLM.
- **Fork this repository and work in your fork,** so we can follow your work. Forks are public.
- **Your system has a frontend:** a working page that shows the architecture of your system as a diagram and runs the system live on rows you give it. Start it however you like: your video shows it running.
- **At 09:00 on 29 Oct we add a sample of 20 rows to this repository,** as one CSV file, `sample/sample.csv`: new rows, in the shape of the run's input with every `label` empty (`id`, `text`, `date`, `label`). The video you hand in at 11:00 shows your frontend running your system live on them, beside its architecture diagram.
- **Your submission is a pull request** from your fork to this repository, one per team, titled with your team's name and opened before 11:00 on 29 Oct. It holds both the `Dockerfile` we run your system from, as [`contracts/README.md`](contracts/README.md) says, and what starts your frontend. What it holds at 11:00 is what we run: a push to it, or closing it, after 11:00 disqualifies the team.
- **We run every team's system** after 11:00, offline, for up to 30 minutes, on about 20,000 rows of the universe, the practice rows among them with their labels. It hands in one table, `/output/output.csv`: a label for each of the roughly 18,000 unlabelled rows and exactly 1,000 new claims, every one true, and we judge both. How it runs, what it reads and what it hands in is in [`contracts/README.md`](contracts/README.md).

## What is here

| Path | What |
| --- | --- |
| `data/practice.csv` | The practice dataset: one row per sentence of the practice part of the universe, each labelled `true`, `false` or `unsure`, with the facts it states |
| `data/lexicon.json` | The subjects the dataset names, with their aliases and attributes, and the events |
| `contracts/` | What you get, how we run your system and what it hands in |
| `sample/sample.csv` | From 09:00 on 29 Oct: the 20 rows of the sample, unlabelled |
| `scripts/check.sh` | Runs your system as we will, on the practice data, and checks its table |

## Check your system

`scripts/check.sh` builds the image from your `Dockerfile` and runs it as we will: offline, as user 10001, on a read-only root, with 2 CPUs and 6 GiB, until it exits or 30 minutes pass. Its input is the practice dataset in the run's shape, with every fifth row's label hidden. It then checks your table's columns, ids, lengths, labels and confidences, that it holds exactly 1,000 new claims, and counts your labels on the hidden rows against the practice labels. It needs Docker and Python 3.

It does not check a claim's text against `claim-forms.md`, its form against `entry.schema.json`, or its names against the lexicon: we check those after the run. And an image that holds `data/practice.csv` knows the hidden labels, so the label count here is an upper bound.

```sh
scripts/check.sh                 # build the image and run it
scripts/check.sh my-image:tag    # run an image you built
MINUTES=5 scripts/check.sh       # a shorter run
```

A run that passes here can still fail ours on what the practice data does not hold.
