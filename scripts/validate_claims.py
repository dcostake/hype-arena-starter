"""Check the claims of a table against contracts/entry.schema.json and the lexicon.

    python scripts/validate_claims.py output.csv [data/lexicon.json]

What check_output.py leaves to the referee: each claim's form against the schema, and every
fact it names (subject, attribute, link, target, event) against the lexicon. Needs jsonschema
(pip install jsonschema), which the system itself does not.
"""
import csv
import json
import os
import sys

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(table: str, lexicon_path: str) -> int:
    entry = json.load(open(os.path.join(ROOT, "contracts", "entry.schema.json"), encoding="utf-8"))
    value = json.load(open(os.path.join(ROOT, "contracts", "claimed_value.schema.json"), encoding="utf-8"))
    registry = Registry().with_resources([(entry["$id"], Resource.from_contents(entry)),
                                          (value["$id"], Resource.from_contents(value))])
    validator = Draft202012Validator(entry, registry=registry)
    lexicon = json.load(open(lexicon_path, encoding="utf-8"))
    names = {"".join(n.split()).lower(): s for s in lexicon["subjects"] for n in [s["name"], *s["aliases"]]}
    events = {"".join(e.split()).lower() for e in lexicon["events"]}

    def has(subject: str, attribute: str) -> bool:
        s = names.get("".join(subject.split()).lower())
        return s is not None and attribute in s["attributes"]

    claims = bad = 0
    with open(table, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not (r.get("type") or "").strip():
                continue
            claims += 1
            form = json.loads(r["form"])
            entry_ = {"id": r["id"], "type": r["type"], "text": r["text"], "form": form, "tag": "true",
                      "confidence": float(r["confidence"])}
            errors = [e.message for e in validator.iter_errors(entry_)]
            if r["type"] == "compare":
                errors += [f"no fact {x['subject']}.{x['attribute']}" for x in (form["left"], form["right"])
                           if not has(x["subject"], x["attribute"])]
            elif r["type"] == "relation":
                errors += [] if has(form["subject"], form["link"]) else ["no such link"]
                errors += [] if has(form["target"], form["attribute"]) else ["no such target fact"]
            else:
                errors += [] if has(form["subject"], form["attribute"]) else ["no such fact"]
            if r["type"] == "at_event" and "".join(form["event"].split()).lower() not in events:
                errors.append("no such event")
            if errors:
                bad += 1
                print(f"  {r['id']}: {'; '.join(errors[:3])}")
    print(f"{claims} claims, {bad} invalid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "lexicon.json")))
