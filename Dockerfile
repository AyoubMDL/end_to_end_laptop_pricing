FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN apt-get install libgomp1

RUN pip install -r requirements.txt

CMD ["python3", "application.py"]
