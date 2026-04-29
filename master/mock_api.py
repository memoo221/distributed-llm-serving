import os

from fastapi import FastAPI

from master.routers.mock_api import router


MASTER_ID = os.getenv("MASTER_ID", "master-unknown")

app = FastAPI(title=f"Mock Master {MASTER_ID}")
app.include_router(router)
