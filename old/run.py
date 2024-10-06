import uvicorn
from dotenv import load_dotenv
from os import getenv, path
import sys

if __name__ == "__main__":
  load_dotenv()
  sys.path.append(path.join(path.dirname(__file__), "src"))
  
  SERVER_HOST = getenv("SERVER_HOST")
  SERVER_PORT = int(getenv("SERVER_PORT"))
  DEV_ENV = bool(getenv("DEV_ENV"))
  
  uvicorn.run("src.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=DEV_ENV)