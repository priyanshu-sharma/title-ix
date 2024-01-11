import logging
import logging.config
from fastapi import FastAPI
from api import api_router
from fastapi.middleware.cors import CORSMiddleware
from server_config import LOGGING_CONFIG_PATH

logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crawler API", version="0.1")
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("Starting Up")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shut Down")
