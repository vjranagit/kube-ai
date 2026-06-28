FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8000
EXPOSE 8000

# Use ENTRYPOINT so that Kubernetes spec.args (e.g. --max-num-seqs=256) are
# appended to the command rather than replacing it entirely.
ENTRYPOINT ["python", "app.py"]
