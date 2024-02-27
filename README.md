# MNIST Observability

A simple web app to manage MNIST experiments and view metrics.

## Prerequisites

- Redis

- MongoDB

## Getting started

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Modify the `.env` file to fill in environment variables.

## API

GET /api/experiments

GET /api/experiments/:id

POST /api/experiments

POST /api/experiments/:id/run

POST /api/experiments/:id/stop

POST /api/experiments/:id/clone

DELETE /api/experiments/:id

## Data model

Abstract:
- id
- created_at
- updated_at

Hyperparam(Abstract):
- epoches
- learning_rate [GS]
- dropout [GS]
- batch_size
- optimizer (Enum: Adam, SGD, RMSprop)
- architecture
  + hidden_layers
  + activation_func
- output_activation_func (Enum: softmax)
- loss_func (MSE, MAE)

Metrics(Abstract):
- loss
- accuracy
- precision
- recall
- f1_score
- runtime

Experiment(Abstract):
- name
- celery_task_id
- use_gpu
- status: [DRAFT, RUNNING, SUCCESS, ERROR]
- deleted: bool
- log_dir
- checkpoint_dir
- hyperparams: Hyperparam
- last_result: Metrics
- result_per_epoch: Metrics[]