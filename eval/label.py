"""
eval/label.py — label the gold set by hand, one row at a time.

    ./venv/bin/python eval/label.py [--gold eval/gold_set.csv]

Shows each unlabeled row (business, merchant, description, amount). Press
1-5 to label it Utilities / Shopping / Dining / Travel/Transportation /
Other, s to skip it for now, q to quit. Each answer is saved immediately, so
quitting (or Ctrl-C) loses nothing, and the next run resumes at the first
row that is still unlabeled; skipped rows come back then.

It never suggests or pre-fills a label. It writes only the label column:
the rest of the file stays byte-for-byte identical (quoting, line endings,
multi-line fields), because only that field of the raw text is replaced,
and each edit is re-parsed and checked before it is saved.
"""
import argparse
import csv
import io
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from afw.prompt_versions import PROMPTS

DEFAULT_PATH = os.path.join(REPO, "eval", "gold_set.csv")
CHOICES = {str(n): c.value for n, c in enumerate(PROMPTS["v2"].categories, start=1)}
MENU = "  ".join(f"{k} {v}" for k, v in CHOICES.items()) + "  s skip  q quit"
LINES = re.compile(r"[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+$")
TERMINATOR = re.compile(r"(\r\n|\n|\r)$")


def read_records(path):
    """-> [(fields, raw)] for every record, where raw is its exact text,
    including quoting, embedded newlines and its line terminator."""
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    consumed = []

    def lines():
        for line in LINES.findall(text):
            consumed.append(line)
            yield line

    records = []
    for fields in csv.reader(lines()):
        records.append((fields, "".join(consumed)))
        consumed.clear()
    if "".join(raw for _, raw in records) != text:
        raise SystemExit(f"{path}: could not split the file into records without losing bytes")
    return records


def with_label(fields, raw, label):
    """The same record with only its last field (the label) replaced."""
    terminator = TERMINATOR.search(raw)
    end = terminator.start() if terminator else len(raw)
    body = raw[:end]
    new_raw = body[:body.rfind(",") + 1] + label + raw[end:]
    if next(csv.reader(io.StringIO(new_raw, newline=""))) != fields[:-1] + [label]:
        raise SystemExit("refusing to save: the edited row would not parse back to the same fields")
    return fields[:-1] + [label], new_raw


def save(path, records):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write("".join(raw for _, raw in records))
    os.replace(tmp, path)


def read_line_key():
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.strip().lower()


def read_single_key():
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if key == "\x03":
        raise KeyboardInterrupt
    if key in ("", "\x04"):
        raise EOFError
    return key.lower()


def run(path, read_key, out=print):
    records = read_records(path)
    header = records[0][0]
    if not header or header[-1] != "label":
        raise SystemExit(f"{path}: the last column must be 'label'")
    column = {name: header.index(name) for name in ("business", "merchant", "description", "amount")}
    total = len(records) - 1
    todo = [i for i in range(1, len(records)) if not records[i][0][-1].strip()]
    if not todo:
        out(f"All {total} rows are labeled.")
        return

    labeled = total - len(todo)
    try:
        for i in todo:
            fields = records[i][0]
            out(f"\nRow {i} of {total} · {labeled} of {total} labeled")
            for name in ("business", "merchant", "description", "amount"):
                out(f"  {name}: {fields[column[name]]}")
            out(MENU)
            while True:
                key = read_key()
                if key in CHOICES:
                    records[i] = with_label(fields, records[i][1], CHOICES[key])
                    save(path, records)
                    labeled += 1
                    out(f"  saved: {CHOICES[key]}")
                    break
                if key == "s":
                    break
                if key == "q":
                    raise EOFError
                out("Press 1-5, s or q.")
    except (EOFError, KeyboardInterrupt):
        pass
    out(f"\n{labeled} of {total} labeled. Run again to continue.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Label the gold set by hand.")
    parser.add_argument("--gold", default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    run(args.gold, read_single_key if sys.stdin.isatty() else read_line_key)


if __name__ == "__main__":
    main()
