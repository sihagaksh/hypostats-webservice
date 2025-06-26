# Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install build tools + deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port (if your app uses a non-default port, change this)
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
