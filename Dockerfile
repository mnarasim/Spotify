
FROM python:3.10


WORKDIR .


RUN apt-get update

RUN apt-get install libsndfile1 -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --no-cache-dir -U -r  requirements.txt



COPY app/ .

EXPOSE 5000

CMD ["python","main.py"]


