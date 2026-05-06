import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from master.master_node import MasterNode
from master.routers.master_router import router
from master.services.config import load_bootstrap_workers
from master.services.forwarder import Forwarder
from master.services.registry import WorkerRegistry


MASTER_ID = os.getenv("MASTER_ID", "master-unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = WorkerRegistry(static_workers=load_bootstrap_workers())
    forwarder = Forwarder()
    master_node = MasterNode(registry=registry, forwarder=forwarder)
    app.state.registry = registry
    app.state.forwarder = forwarder
    app.state.master_node = master_node

    master_node.start()
    yield
    await master_node.stop()
    await forwarder.close()


app = FastAPI(title=f"Master {MASTER_ID}", lifespan=lifespan)
app.include_router(router)
