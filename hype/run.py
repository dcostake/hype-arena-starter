"""Runs the system and writes a table in the shape of the practice data.

The table has the practice columns, `id, text, date, label, form`: every empty
label filled by the system, and in `form` the facts the system reads from each
row. It opens in Excel as it is.

    python -m hype.run data/practice.csv                  # -> results.csv
    python -m hype.run my_rows.csv --out labelled.csv
    python -m hype.run data/practice.csv --relabel        # label every row, ignoring the labels it has
    python -m hype.run my_rows.csv --details              # add confidence, decided_by and reasons

    python -m hype.run --write                            # write sentences, get each label, then the table
    python -m hype.run --write --table my_rows.csv        # ... added to the rows of a table

In the arena container, with no table given, it reads /universe/ and writes the
contract's /output/output.csv instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date as Date

from .characterize import DEFAULT_LEXICON
from .pipeline import HypePipeline, Reading, Verdict

PRACTICE_COLUMNS = ["id", "text", "date", "label", "confidence", "form", "residue"]
DETAIL_COLUMNS = ["decided_by", "reasons"]
CONTRACT_COLUMNS = ["id", "type", "text", "form", "label", "confidence"]
ARENA_INPUT = "/universe/universe.csv"
ARENA_LEXICON = "/universe/lexicon.json"
ARENA_OUTPUT = "/output/output.csv"


# --- tables ------------------------------------------------------------------------------

def normalize_date(text: str) -> str | None:
    """A date as YYYY-MM-DD, from the ways a table writes one: 2026-10-29, 2026/10/29, 2026-10-29T08:00,
    and, as Excel writes it in Romania, 29.10.2026 or 29/10/2026 (day first). None if it is no date."""
    text = (text or "").strip()
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T ].*)?$", text)
    if m:
        y, mo, d = m.groups()
    else:
        m = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?:[T ].*)?$", text)
        if not m:
            return None
        d, mo, y = m.groups()
    try:
        return Date(int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return None


def read_table(path: str) -> list[dict]:
    """The rows of a CSV with `id`, `text` and `date`; a semicolon CSV, as Excel writes it in Romanian, reads too.
    Dates are brought to YYYY-MM-DD; a row whose date is no date keeps it, and `bad_date` marks it."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        data = f.read()
    first = data.split("\n", 1)[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(data.splitlines(), delimiter=delimiter)
    rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in reader]
    missing = {"id", "text", "date"} - {(c or "").strip().lower() for c in reader.fieldnames or []}
    if missing:
        raise SystemExit(f"{path}: the table has no column {', '.join(sorted(missing))}")
    rows = [r for r in rows if r.get("id")]
    for r in rows:
        r["label"] = (r.get("label") or "").strip().lower()
        iso = normalize_date(r["date"])
        if iso:
            r["date"] = iso
        else:
            r["bad_date"] = True
    bad = [r["id"] for r in rows if r.get("bad_date")]
    if bad:
        print(f"{len(bad)} rows have a date that is not a date and are labelled unsure: {', '.join(bad[:5])}"
              + (" ..." if len(bad) > 5 else ""), file=sys.stderr)
    return rows


def readable(rows: list[dict]) -> list[dict]:
    """The rows the system can read: a text and a real date."""
    return [r for r in rows if not r.get("bad_date") and r.get("text")]


def write_table(path: str, columns: list[str], rows: list[dict], excel: bool = True) -> None:
    """Write under another name, then rename, so a reader never sees half a table.
    `excel` adds a byte-order mark, so Excel reads the euro sign and the accents."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8-sig" if excel else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def as_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def practice_rows(rows: list[dict], verdicts: dict[str, Verdict], readings: dict[str, Reading]) -> list[dict]:
    """The rows in the practice shape: the system's label where the row had none, and on every row
    the system's form and the residue the form leaves (values, names and references it does not take)."""
    out = []
    for r in rows:
        v = verdicts.get(r["id"])
        reading = readings.get(r["id"])
        row = {"id": r["id"], "text": r["text"], "date": r["date"],
               "label": v.label if v else r.get("label", ""),
               "form": as_json(reading.form if reading else []),
               "residue": as_json(reading.residue if reading else {})}
        if v:
            row["confidence"] = "" if v.confidence is None else f"{v.confidence:.4f}"
            row["decided_by"] = v.decided_by
            row["reasons"] = " | ".join(f"{f.name}: {f.readings[0] if f.readings else 'no reading'}" for f in v.facts) or "no fact read"
        elif not r.get("label"):
            # a row the system could not read: unsure, which takes no confidence
            row["label"], row["confidence"] = "unsure", ""
            row["decided_by"] = "unreadable date" if r.get("bad_date") else "not labelled"
        else:
            row["confidence"] = ""  # the input's own label: the system has no confidence in it to give
            row["decided_by"] = "input"
        out.append(row)
    return out


def label_table(pipeline: HypePipeline, rows: list[dict], relabel: bool = False, same_page: bool = True):
    if relabel:
        rows = [{**r, "label": ""} for r in rows]
    verdicts, readings = pipeline.analyse(readable(rows), same_page=same_page)
    return rows, {v.id: v for v in verdicts}, readings


def table_columns(path: str) -> list[str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        first = f.readline()
    delimiter = ";" if first.count(";") > first.count(",") else ","
    return [c.strip().lower() for c in next(csv.reader([first], delimiter=delimiter))]


def parse_sentence(line: str, date: str | None = None) -> tuple[str, str]:
    """`YYYY-MM-DD | sentence`, or a bare sentence that keeps `date`."""
    if "|" in line:
        date, line = (p.strip() for p in line.split("|", 1))
    if not date or len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise ValueError("write the sentence as  YYYY-MM-DD | sentence")
    return date, line.strip()


# --- the function: a sentence in, its label out, the table updated -----------------------------

def add_sentences(sentences: list[tuple[str, str]], table: str = "results.csv",
                  pipeline: HypePipeline | None = None, details: bool = False) -> list[tuple[dict, Verdict]]:
    """Label each (date, sentence) and add it to the end of `table`, which is created if it is not there.

    Each sentence is judged on its own date against everything the system knows (the practice
    rows), not against the other rows of the table, whose labels are the system's own. The
    table is rewritten whole each time, under another name first, so it is never half written.

        from hype.run import add_sentences
        [(row, verdict)] = add_sentences([("2026-04-01", "Hooly AI Pro costs €59 per month.")])
        print(row["label"], verdict.confidence)
    """
    pipeline = pipeline or HypePipeline.load()
    existing = read_table(table) if os.path.exists(table) else []
    columns = table_columns(table) if existing else PRACTICE_COLUMNS + (DETAIL_COLUMNS if details else [])
    for c in PRACTICE_COLUMNS + (DETAIL_COLUMNS if details else []):
        if c not in columns:
            columns.append(c)
    taken = {r["id"] for r in existing}
    n = 1
    rows = []
    for date, text in sentences:
        while f"w-{n}" in taken:
            n += 1
        taken.add(f"w-{n}")
        rows.append({"id": f"w-{n}", "text": text, "date": date, "label": ""})
    verdicts, readings = pipeline.analyse(rows, same_page=False)
    by_id = {v.id: v for v in verdicts}
    added = practice_rows(rows, by_id, readings)
    write_table(table, columns, existing + added)
    return [(row, by_id[row["id"]]) for row in added]


def print_verdict(row: dict, v: Verdict) -> None:
    conf = "" if v.confidence is None else f" {v.confidence:.2f}"
    print(f"  {row['id']}  -> {v.label.upper()}{conf}   ({v.decided_by})")
    for f in v.facts:
        print(f"     {f.name}   holds {f.holds:.2f}" + (f"   [{f.readings[0]}]" if f.readings else ""))
    if not v.facts:
        print("     no fact read: the label rests on the model's prior")
    if row.get("residue") and row["residue"] != "{}":
        print(f"     residue: {row['residue']}")


# --- the ways to run -------------------------------------------------------------------------

def run_table(args) -> int:
    started = time.time()
    rows = read_table(args.input)
    original = {r["id"]: r.get("label", "") for r in rows}
    relabel = args.relabel
    if not relabel and rows and all(r.get("label") for r in rows):
        relabel = True
        print("every row has a label already: labelling every row, and comparing with the labels it had")
    pipeline = HypePipeline.load(args.lexicon or DEFAULT_LEXICON)
    rows, verdicts, readings = label_table(pipeline, rows, relabel=relabel, same_page=not args.each_row_alone)
    columns = PRACTICE_COLUMNS + (DETAIL_COLUMNS if args.details else [])
    write_table(args.out, columns, practice_rows(rows, verdicts, readings))
    counts = {k: sum(1 for v in verdicts.values() if v.label == k) for k in ("true", "false", "unsure")}
    print(f"{len(rows)} rows, {len(verdicts)} labelled by the system: "
          f"{counts['true']} true, {counts['false']} false, {counts['unsure']} unsure, in {time.time() - started:.1f} s")
    scored = [(v.label, original[i]) for i, v in verdicts.items() if original.get(i)]
    if scored:
        right = sum(p == t for p, t in scored)
        print(f"against the labels the table had: {right} of {len(scored)} right ({right / len(scored):.1%})"
              + ("; the practice rows are in the system's knowledge, so this is an upper bound" if relabel else ""))
    print(f"wrote {os.path.abspath(args.out)}")
    return 0


def run_add(args) -> int:
    """Each --add sentence labelled and added to the end of the --out table."""
    try:
        sentences = []
        date = args.date
        for line in args.add:
            date, text = parse_sentence(line, date)
            sentences.append((date, text))
    except ValueError as e:
        print(e)
        return 1
    pipeline = HypePipeline.load(args.lexicon or DEFAULT_LEXICON)
    for row, v in add_sentences(sentences, args.out, pipeline, details=args.details):
        print_verdict(row, v)
    print(f"added {len(sentences)} to {os.path.abspath(args.out)}")
    return 0


def run_write(args) -> int:
    """Read sentences from the keyboard; label each at once and add it to the table at once."""
    pipeline = HypePipeline.load(args.lexicon or DEFAULT_LEXICON)
    if args.table:
        rows, verdicts, readings = label_table(pipeline, read_table(args.table))
        write_table(args.out, PRACTICE_COLUMNS + (DETAIL_COLUMNS if args.details else []), practice_rows(rows, verdicts, readings))
        print(f"labelled {args.table} into {os.path.abspath(args.out)}; your sentences go after its rows")
    print(f"Each sentence is labelled and added to {os.path.abspath(args.out)} at once.")
    print("Write  YYYY-MM-DD | sentence  , or just the sentence to keep the last date.")
    print("'why' explains the last label; 'done' or an empty line stops.\n")
    date, last = args.date, None
    while True:
        try:
            line = input(f"[{date or 'no date'}] > ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("done", "q", "quit", "exit"):
            break
        if line.lower() == "why":
            print(last.explain() if last else "nothing labelled yet")
            continue
        try:
            date, text = parse_sentence(line, date)
        except ValueError as e:
            print(f"  {e}")
            continue
        [(row, last)] = add_sentences([(date, text)], args.out, pipeline, details=args.details)
        print_verdict(row, last)
    print(f"\nthe table is {os.path.abspath(args.out)}")
    return 0


def run_arena(input_path: str = ARENA_INPUT, lexicon_path: str | None = None, output_path: str = ARENA_OUTPUT) -> int:
    """The contract's run: a label for every unlabelled input row, and exactly 1,000 new true claims.

    The table is written three times, each complete: at once with every row unsure, then with the
    claims, then with the labels. A run stopped at the deadline still hands in the best table so far.
    """
    from .claims import claim_rows, generate_claims

    started = time.time()
    rows = read_table(input_path)
    unlabelled = [r for r in rows if not r.get("label")]
    lexicon = lexicon_path or (ARENA_LEXICON if os.path.exists(ARENA_LEXICON) else DEFAULT_LEXICON)

    def label_rows(verdicts: dict) -> list[dict]:
        out = []
        for r in unlabelled:
            v = verdicts.get(r["id"])
            label = v.label if v else "unsure"
            conf = "" if label == "unsure" or v.confidence is None else f"{v.confidence:.4f}"
            out.append({"id": r["id"], "type": "", "text": "", "form": "", "label": label, "confidence": conf})
        return out

    # first the labels of the input's unlabelled rows, in the input's order, then the 1,000 claims
    write_table(output_path, CONTRACT_COLUMNS, label_rows({}), excel=False)
    claims_out: list[dict] = []
    try:
        claims, _ = generate_claims(lexicon, readable(rows))
        taken = {r["id"] for r in rows}
        claims_out = [c for c in claim_rows(claims) if c["id"] not in taken]
        write_table(output_path, CONTRACT_COLUMNS, label_rows({}) + claims_out, excel=False)
        print(f"{len(claims_out)} claims in {time.time() - started:.0f} s", flush=True)
    except Exception as e:  # the labels are still worth handing in
        print(f"the claims failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    verdicts = label_safely(HypePipeline.load(lexicon), readable(rows))
    write_table(output_path, CONTRACT_COLUMNS, label_rows(verdicts) + claims_out, excel=False)
    labelled = sum(1 for r in unlabelled if r["id"] in verdicts)
    print(f"labelled {labelled} of {len(unlabelled)} rows, then {len(claims_out)} claims, into {output_path} "
          f"in {time.time() - started:.0f} s")
    return 0


def label_safely(pipeline: HypePipeline, rows: list[dict]) -> dict[str, Verdict]:
    """The verdicts of the whole table; if that fails, page by page, so one page that breaks the
    system leaves only its own rows unsure."""
    try:
        return {v.id: v for v in pipeline.label(rows)}
    except Exception as e:
        print(f"labelling the whole table failed ({type(e).__name__}: {e}); labelling page by page",
              file=sys.stderr, flush=True)
    from .resolve import segment_pages
    labelled_rows = [r for r in rows if r.get("label")]
    verdicts: dict[str, Verdict] = {}
    for page in segment_pages(rows):
        if all(r.get("label") for r in page):
            continue
        ids = {r["id"] for r in page}
        try:
            for v in pipeline.label([r for r in labelled_rows if r["id"] not in ids] + page):
                verdicts[v.id] = v
        except Exception as e:
            print(f"  page of {page[0]['date']} ({len(page)} rows) stays unsure: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
    return verdicts


def utf8_streams() -> None:
    """Read and write UTF-8 whatever the console's code page, so the euro sign and curly quotes arrive intact."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not sys.stdin.isatty():
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str]) -> int:
    utf8_streams()
    parser = argparse.ArgumentParser(description="Label rows about Hooly and write a table like the practice data.")
    parser.add_argument("input", nargs="?", help="a CSV with id, text, date and label")
    parser.add_argument("--out", default="results.csv", help="the table to write or add to (default: results.csv)")
    parser.add_argument("--relabel", action="store_true", help="label every row, also those that have a label")
    parser.add_argument("--details", action="store_true", help="add the columns confidence, decided_by and reasons")
    parser.add_argument("--each-row-alone", action="store_true", help="do not read consecutive rows of one date as a page")
    parser.add_argument("--add", nargs="+", metavar="'DATE | SENTENCE'", help="label these sentences and add them to the --out table")
    parser.add_argument("--write", action="store_true", help="write sentences one by one; each is labelled and added to the table")
    parser.add_argument("--table", help="with --write: label this table first into --out, then add the sentences after it")
    parser.add_argument("--date", help="with --add or --write: the date of sentences written without one")
    parser.add_argument("--lexicon", help="the lexicon to read names with (default: data/lexicon.json)")
    parser.add_argument("--contract", metavar="UNIVERSE_CSV",
                        help="the arena run on this input, here: labels and 1,000 claims into --out (default output.csv)")
    args = parser.parse_args(argv)
    if args.add:
        return run_add(args)
    if args.write:
        return run_write(args)
    if args.input:
        return run_table(args)
    if args.contract:
        return run_arena(args.contract, args.lexicon, args.out if args.out != "results.csv" else "output.csv")
    if os.path.exists(ARENA_INPUT):
        return run_arena()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
