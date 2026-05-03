import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from master.routers.master_router import router
from master.services.config import load_bootstrap_workers
from master.services.forwarder import Forwarder
from master.services.registry import WorkerRegistry


MASTER_ID = os.getenv("MASTER_ID", "master-unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = WorkerRegistry(static_workers=load_bootstrap_workers())
    forwarder = Forwarder()
    app.state.registry = registry
    app.state.forwarder = forwarder
    yield
    await forwarder.close()


app = FastAPI(title=f"Master {MASTER_ID}", lifespan=lifespan)
app.include_router(router)
