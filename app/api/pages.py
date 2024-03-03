from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")
router = APIRouter()

@router.get('/')
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@router.get('/create')
def create(request: Request):
    return templates.TemplateResponse("create.html", {"request": request})