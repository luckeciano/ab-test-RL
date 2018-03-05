FROM ubuntu:latest
MAINTAINER Luckeciano Melo "luckeciano.melo@vtex.com.br"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["web_server.py"]