FROM postgis/postgis
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-16
RUN cd /tmp && rm -rf pgvector && git clone --branch v0.6.2 https://github.com/pgvector/pgvector.git
RUN cd /tmp/pgvector && make && make install
# then run the command to start the database
