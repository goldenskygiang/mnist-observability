from app.celery import celeryapp

@celeryapp.task
def start_experiment():
    pass

@celeryapp.task
def stop_experiment():
    pass

def get_job_status():
    pass