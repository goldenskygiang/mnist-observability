from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.api.endpoints.experiments import get_experiment_with_id
from app.models.experiment import ExperimentModel
from app.models.metrics import Metrics

templates = Jinja2Templates(directory="app/templates")
router = APIRouter()

@router.get('/')
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html"
    )

@router.get('/details/{id}')
async def details(request: Request, id: str):
    exp = await get_experiment_with_id(id)
    exp = ExperimentModel.model_dump(ExperimentModel.model_validate(exp))
    
    return templates.TemplateResponse(
        request=request,
        name="details.html",
        context=exp
    )

@router.get('/create')
def create(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="create.html"
    )