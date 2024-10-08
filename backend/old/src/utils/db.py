from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from os import getenv
from utils.print_log import log_success, log_error

from utils.model_list import model_list

load_dotenv()
MONGODB_CLUSTER_URI = getenv("MONGODB_CLUSTER_URI")
DATABASE_NAME = getenv("DATABASE_NAME")

client = AsyncIOMotorClient(MONGODB_CLUSTER_URI)

async def init_db():
  database = client[DATABASE_NAME]
  
  try:
    await init_beanie(database, document_models=model_list)
    log_success("Connected with MongoDB\n")
  except Exception as error:
    log_error("Error connecting with MongoDB", error)
    
async def disconnect_db():
  client.close()
  log_success("Disconnected from MongoDB\n")
