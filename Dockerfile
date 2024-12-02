FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
CMD python app.py 2> errors.txt