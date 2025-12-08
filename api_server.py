from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from eco_simu.service import GraphRunRequest, GraphRunResult, run_graph

app = FastAPI(title="eco_simu Agent Graph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run", response_model=GraphRunResult)
def run(req: GraphRunRequest):
    return run_graph(req)


# uvicorn api_server:app --port 8000
