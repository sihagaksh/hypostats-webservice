# 1️⃣ Base image: Python 3.9
FROM python:3.9-slim

# 2️⃣ Create & switch to app directory
WORKDIR /app

# 3️⃣ Copy requirements & install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# 4️⃣ Copy all your source files
COPY . .

# 5️⃣ Expose the default Flask port (not strictly needed, but docs do)
EXPOSE 5000

# 6️⃣ Launch via Gunicorn, binding to Render’s $PORT
CMD gunicorn app:app --bind 0.0.0.0:$PORT
