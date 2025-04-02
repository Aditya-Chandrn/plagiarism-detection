import motor.motor_asyncio
from dotenv import dotenv_values

config = dotenv_values(".env")

client = motor.motor_asyncio.AsyncIOMotorClient(config["MONGODB_CLUSTER_URI"])

database = client[config["DATABASE_NAME"]]

user_collection = database["users"]
document_collection = database["documents"]
submission_collection = database["submissions"]

print("Connected to the MongoDB!")
