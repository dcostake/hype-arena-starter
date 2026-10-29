# Kolmogorov team 56: the Hype Arena system.
# It reads /universe/universe.csv and /universe/lexicon.json, and writes /output/output.csv
# (contracts/README.md). It needs only the Python standard library: nothing from the network,
# at build time or at run time, beyond this base image.
FROM python:3.14-slim

# The run has no home directory and a read-only root: only /tmp and /output are writable.
# So the home and every cache point at /tmp, and Python writes no bytecode at run time.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1

WORKDIR /app

# The system, its learned models (hype/model/), and the practice data it knows, forms included.
COPY hype/ hype/
COPY data/ data/

# Compile once here, so the read-only run starts on bytecode, then run the system on the practice
# data: a broken import or a failing run stops the build here rather than the arena run.
RUN python -m compileall -q hype \
 && python -m hype.run --contract data/practice.csv --out /tmp/build-check.csv \
 && python -c "import csv; rows = list(csv.DictReader(open('/tmp/build-check.csv', encoding='utf-8'))); n = sum(1 for r in rows if r['type']); assert n == 1000, f'{n} claims, not 1000'" \
 && rm -f /tmp/build-check.csv

CMD ["python", "-m", "hype.run"]

# Every run starts as this user (contracts/README.md): the Dockerfile ends with it.
USER 10001
