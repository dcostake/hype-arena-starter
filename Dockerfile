# The Hype Arena system: reads /universe/, writes /output/output.csv (contracts/README.md).
# Only the Python standard library: the image needs nothing from the network at run time.
FROM python:3.14-slim

# no home directory, a read-only root: nothing writes outside /tmp and /output
ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app
# the system, its learned models, and the practice data it knows (forms included)
COPY hype/ hype/
COPY data/ data/

USER 10001
CMD ["python", "-m", "hype.run"]
