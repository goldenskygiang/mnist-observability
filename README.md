# MNIST Observability

A simple web app to manage MNIST Handwritten Digit Classification experiments.

## Table of Contents

- [MNIST Observability](#mnist-observability)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Getting started](#getting-started)
    - [Run locally](#run-locally)
    - [Use with Docker](#use-with-docker)
  - [Features](#features)
    - [Manage the list of experiments and sort them by criteria](#manage-the-list-of-experiments-and-sort-them-by-criteria)
    - [Create a new experiment with hyperparameters - and even grid-search on learning rates \& batch sizes](#create-a-new-experiment-with-hyperparameters---and-even-grid-search-on-learning-rates--batch-sizes)
    - [Monitor the status of experiment with live log streaming](#monitor-the-status-of-experiment-with-live-log-streaming)
    - [View the experiment results with rich visualization](#view-the-experiment-results-with-rich-visualization)
    - [Leverage more information from API endpoints](#leverage-more-information-from-api-endpoints)
  - [Project structure](#project-structure)
  - [Limitations \& Known issues](#limitations--known-issues)

## Prerequisites

In order to run this project, you need to have Redis and MongoDB servers installed.

This project was developed on Ubuntu 22.04, and has not been tested with Windows. Some Python dependencies may not work on Windows.

## Getting started

### Run locally

Create a Python environment then install all dependencies.

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the two Redis and MongoDB servers. Alternatively, you can start Docker containers of Redis and MongoDB.

Create an `.env` file to fill in environment variables, especially the connection strings of Redis and MongoDB.

Start the web server with the following command, and the website should be up at [127.0.0.1:8080](http://127.0.0.1:8080).

```sh
uvicorn app.main:app
```

Open another terminal instance and start Celery worker with this command:
```sh
celery -A app worker --loglevel=INFO -P [threads/solo/gevent]
```

In the `-P` flag, please choose one of the three values above, which will instruct `celery` to use the specified task pool.

### Use with Docker

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

If you do not make any modifications to the Docker configurations, you should be able to create and start 6 Docker containers (MongoDB, Redis, web, Celery Flower, and 2 Celery workers).

- The main website should be up at [localhost:8000](http://localhost:8000).

- In addition, you can observe Celery workers' statuses via Flower, exposed at [localhost:5555](http://localhost:5555).

## Features

### Manage the list of experiments and sort them by criteria

![](img/feat1.png)

### Create a new experiment with hyperparameters - and even grid-search on learning rates & batch sizes

Customize every experiment by adjusting hyperparameters: number of epochs, optimizer, activation function.

Have multiple learning rates or batch sizes to try? Input them and click once to run all combinations!

![](img/feat2.png)

You can even make your experiment reproducible with seed - every experiment with the same settings and seed value will behave the same!

### Monitor the status of experiment with live log streaming

Observe the experiment with live log streaming.

![](img/feat3-1.jpeg)

And find out the issue if your model fail!

![](img/feat3-2.jpeg)

Wait for too long? Just turn off the browser (but not the Celery workers) and do other things. You can come back later at any time and still see the full log!

### View the experiment results with rich visualization

View how the curves of metrics behave in each grid-search configuration during training, and how different grid-search combinations compare with each other in the testing phase.

![](img/feat4.jpeg)

### Leverage more information from API endpoints

![](img/feat5.png)

Access the information programmatically via API endpoints. Swagger doc is available at [127.0.0.1:8080/docs](http://127.0.0.1:8080/docs) or [localhost:8000/docs](http://localhost:8000/docs) if using Docker.

## Project structure

This project uses MongoDB - a NoSQL database - to store information of the experiments, Celery as a distributed task queue to run model training and testing, and Redis as a message broker between the web application and Celery workers. The clean-looking frontend is designed with Bootstrap, with chart elements powered by Chart.js and log viewer by Ace code editor.

Below is the folder structure and some key components of this project.

```
.
├── app
│   ├── api
│   │   ├── endpoints
│   │   │   └── experiments.py
│   │   ├── pages.py
│   │   └── router.py
│   ├── celery.py
│   ├── config.py
│   ├── __init__.py
│   ├── main.py
│   ├── mnist
│   │   └── ...
│   ├── models
│   │   └── ...
│   ├── services
│   │   └── ...
│   ├── static
│   │   └── ...
│   └── templates
│       └── ...
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
└── sample.env
```

- `app` folder: Contains the whole FastAPI application source code.
- `app/config.py`: Contains and loads the configuration of the application from environment variables.
- `app/api/endpoints/`: Contains the backend logic of API endpoints.
- `app/api/pages.py`: Handles server-side rendering of pages with templates.
- `app/mnist/`: Handles MNIST datasets and PyTorch model initialization, training and testing.
- `app/models/`: Contains the MongoDB data schemas (not to be confused with the MNIST model).
- `app/services/`: Contains the logic of the Celery worker.
- `app/static/`: Contains static content such as JavaScript/CSS files from third-party libraries.
- `app/templates/`: Contains Jinja2 HTML templates.

## Limitations & Known issues

- For an experiment with more than one grid-search configurations, the homepage will select the one with **the highest F-1 Score in the testing phase** to display on the home page. This will be also designated as the best grid-search configuration in the Details page.
- The maximum number of grid-search configurations is 16. It is the number of learning rates multiplied with the number of batch sizes. Hence, you cannot add more than 16 learning rates or batch sizes on the Create page.
- The app cannot resume jobs if their Celery workers shut down unexpectedly while running (e.g. Ctrl+C the worker).
- Omitting the `-P` flag when starting Celery workers will cause them to hang during model inference. This is due to the fact that Celery uses prefork pool by default, [which may not work well with PyTorch](https://stackoverflow.com/a/63684465).
- The `use_gpu` option in creating new experiments is available via the backend API but not the front-facing web app, since the author does not have a GPU to test. Therefore, the default Docker settings in this project are not configured to support GPU.
    - To show the GPU option in the web app, simply uncommment this block of code in the `app/templates/create.html` file and rerun the project (or rebuild the Docker image `mnist-observability`):

    ```html
    <!-- <div class="row mb-3">
        <label for="useGPU" class="col-md-2">Use GPU</label>
        <div class="col-md-5">
            <div class="form-check">
                <input name="use_gpu" class="form-check-input" type="checkbox" id="useGPU">
            </div>
        </div>
    </div> -->
    ```