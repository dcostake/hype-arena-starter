"""The system's frontend: upload a table, watch it go through each step, take the tables back.

    python -m hype.server            # then open http://localhost:8000
    python -m hype.server --port 8080

One page (`hype/web/index.html`) and three calls, all on the Python standard library:

- `POST /api/run` with a CSV in the practice shape (`id, text, date`, and `label`
  where known) streams one JSON line per step as it ends, then the result: a
  preview of every labelled row with its reasons, and three tables as CSV text
  (the table in the practice shape with form and residue, the contract's
  `output.csv`, and the claims).
- `GET /api/sample` answers `sample/sample.csv`, when the repository has it.
- `GET /api/health` answers whether the system is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .characterize import DEFAULT_LEXICON, ROOT
from .claims import CLAIM_TYPES, claim_rows, generate_claims
from .pipeline import HypePipeline
from .run import CONTRACT_COLUMNS, DETAIL_COLUMNS, PRACTICE_COLUMNS, practice_rows

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SAMPLE = os.path.join(ROOT, "sample", "sample.csv")
MAX_BODY = 64 * 1024 * 1024


def parse_csv(text: str) -> list[dict]:
    """A CSV in the practice shape; a semicolon CSV, as Excel writes it in Romanian, reads too."""
    text = text.lstrip("﻿")
    first = text.split("\n", 1)[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    columns = {(c or "").strip().lower() for c in reader.fieldnames or []}
    missing = {"id", "text", "date"} - columns
    if missing:
        raise ValueError(f"the table has no column {', '.join(sorted(missing))}; it needs id, text and date")
    rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in reader]
    rows = [r for r in rows if r.get("id") and r.get("text")]
    if not rows:
        raise ValueError("the table has no rows")
    for r in rows:
        d = r.get("date", "")
        if len(d) != 10 or d[4] != "-" or d[7] != "-":
            raise ValueError(f"row {r['id']}: the date {d!r} is not YYYY-MM-DD")
    return rows


def to_csv(columns: list[str], rows: list[dict]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


class Handler(BaseHTTPRequestHandler):
    pipeline: HypePipeline
    lexicon_path: str = DEFAULT_LEXICON
    lock = threading.Lock()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.command} {self.path}\n")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(WEB, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/api/sample":
            if not os.path.exists(SAMPLE):
                return self._json(404, {"error": "sample/sample.csv is not in the repository yet"})
            with open(SAMPLE, "rb") as f:
                return self._send(200, f.read(), "text/csv; charset=utf-8")
        if self.path == "/api/health":
            return self._json(200, {"ok": True, "sample": os.path.exists(SAMPLE)})
        return self._json(404, {"error": "no such page"})

    def do_POST(self):
        if self.path != "/api/run":
            return self._json(404, {"error": "no such call"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "the table is over 64 MB"})
        try:
            rows = parse_csv(self.rfile.read(length).decode("utf-8", errors="replace"))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        with_claims = "claims=0" not in (self.headers.get("X-Options") or "")
        # one JSON line per step as it ends; the connection closes when the result is sent
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with self.lock:
            self._run(rows, with_claims)

    def _emit(self, data: dict) -> None:
        self.wfile.write((json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _run(self, rows: list[dict], with_claims: bool) -> None:
        started = time.time()
        last = [started]

        def step(stage: str, details: dict) -> None:
            now = time.time()
            self._emit({"stage": stage, "details": details, "seconds": round(now - last[0], 2)})
            if stage != "reading":
                last[0] = now

        try:
            verdicts, readings = self.pipeline.analyse(rows, progress=step)
            by_id = {v.id: v for v in verdicts}
            claims_out = []
            if with_claims:
                claims, gen = generate_claims(self.lexicon_path, rows)
                taken = {r["id"] for r in rows}
                claims_out = [c for c in claim_rows(claims) if c["id"] not in taken]
                step("claims", {"claims": len(claims_out),
                                "by_type": {t: sum(1 for c in claims_out if c["type"] == t) for t in CLAIM_TYPES}})
            table = practice_rows(rows, by_id, readings)
            contract = [{"id": r["id"], "type": "", "text": "", "form": "",
                         "label": by_id[r["id"]].label,
                         "confidence": "" if by_id[r["id"]].confidence is None else f"{by_id[r['id']].confidence:.4f}"}
                        for r in rows if r["id"] in by_id] + claims_out
            step("written", {"rows": len(table), "contract_rows": len(contract)})
            preview = []
            for r in table:
                v = by_id.get(r["id"])
                preview.append({
                    "id": r["id"], "date": r["date"], "text": r["text"], "label": r["label"],
                    "given": v is None, "confidence": None if v is None else v.confidence,
                    "decided_by": "input" if v is None else v.decided_by,
                    "facts": [] if v is None else [{"fact": f.name, "holds": round(f.holds, 3), "read": round(f.reading, 3),
                                                    "reason": f.readings[0] if f.readings else "", "before": f.before,
                                                    "after": f.after} for f in v.facts],
                    "form": readings[r["id"]].form, "residue": readings[r["id"]].residue,
                })
            self._emit({"result": {
                "seconds": round(time.time() - started, 1),
                "preview": preview,
                "claims_preview": claims_out[:40],
                "files": {
                    "labelled.csv": to_csv(PRACTICE_COLUMNS + DETAIL_COLUMNS, table),
                    "output.csv": to_csv(CONTRACT_COLUMNS, contract),
                    "claims.csv": to_csv(CONTRACT_COLUMNS, claims_out) if claims_out else "",
                },
            }})
        except Exception as e:  # the page shows what broke, and the server keeps running
            self._emit({"error": f"{type(e).__name__}: {e}"})


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Serve the system's frontend.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--lexicon", default=DEFAULT_LEXICON, help="the lexicon to read names with")
    args = parser.parse_args(argv)
    print("loading the system...", file=sys.stderr, flush=True)
    Handler.pipeline = HypePipeline.load(args.lexicon)
    Handler.lexicon_path = args.lexicon
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"open http://localhost:{args.port}", file=sys.stderr, flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
