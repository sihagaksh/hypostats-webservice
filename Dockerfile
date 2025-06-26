# ─────────────────────────────────────────────────────────────────
# Use a stable, supported Python version
FROM python:3.9-slim

# Where our code will live
WORKDIR /app

# Copy only requirements first (to leverage Docker cache)
COPY requirements.txt .

# Upgrade pip + build tools, then install deps
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy the rest of your source code
COPY . .

# Expose the port gunicorn (or your Flask) will use
EXPOSE 5000

# Use gunicorn to run your Flask app defined in app.py
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
