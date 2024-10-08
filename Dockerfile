FROM python:3.12.2
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip  # Update pip
RUN pip install -r requirements.txt
EXPOSE $PORT
# CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
CMD gunicorn --workers=1 --timeout 120 --bind 0.0.0.0:$PORT app:app
