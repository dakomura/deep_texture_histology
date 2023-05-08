FROM python:3

WORKDIR /usr/src/app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN pip install --no-cache-dir histo-patch

COPY . .

CMD [ "histopatch" ]
