from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
try:
    from .router import route
except ImportError:
    from router import route

app=FastAPI(title="Support Ticket Routing API",version="1.0.0")
MODEL=Path("artifacts/router.joblib")
class Ticket(BaseModel): text:str=Field(min_length=2,max_length=10000)
@app.get("/health")
def health(): return {"status":"ok","model_loaded":MODEL.exists()}
@app.post("/route")
def route_ticket(ticket:Ticket):
    if not MODEL.exists(): raise HTTPException(status_code=503,detail="Train first: python src/router.py")
    return route(joblib.load(MODEL),ticket.text)
