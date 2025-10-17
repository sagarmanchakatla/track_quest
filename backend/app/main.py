from fastapi import FastAPI
from app.routes import auth
from app.routes import runs
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Territory Fitness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing — tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(runs.router)

@app.get("/")
def root():
    return {"message": "Backend running!"}

