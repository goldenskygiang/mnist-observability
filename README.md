# MNIST Observability

A simple web app to manage MNIST experiments and view metrics.

## Prerequisites

- Redis

- MongoDB

## Getting started

### Run locally

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Modify the `.env` file to fill in environment variables.

Start the two Redis and MongoDB servers.

Start the web server:

```sh
uvicorn app.main:app --reload
```

Start Celery worker:
```sh
celery -A app worker --loglevel=INFO -P [threads/solo/gevent]
```

### The Docker way

Create a new file `docker.env` containing the necessary environment variables. The following one should provide a decent starting point:

```
MONGODB_URL=mongodb://mongodb:27017/
DB_NAME=mnist-observability
REDIS_URL=redis://default:redispw@redis:6379
MNIST_DATASET_DIR=./data
LOG_DIR=./logs
FLOWER_UNAUTHENTICATED_API=true
```

Run `docker compose` commands:

```sh
docker compose build
docker compose up -d
```

## API

GET /api/experiments

GET /api/experiments/:id

POST /api/experiments

POST /api/experiments/:id/run

POST /api/experiments/:id/stop

POST /api/experiments/:id/clone

DELETE /api/experiments/:id
