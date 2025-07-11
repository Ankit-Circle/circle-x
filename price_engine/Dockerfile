FROM python:3.11-bullseye

WORKDIR /app

# Pin pip version (fix known issues with latest pip)
RUN pip install --upgrade pip

# Install all requirements
COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . .

CMD ["gunicorn", "price_engine_backend.wsgi:application", "--bind", "0.0.0.0:8000"]
