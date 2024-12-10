from fastapi import FastAPI, status
from routers import user, document
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from nlp.nltk_setup import download_nltk_data

app = FastAPI(debug=True)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

download_nltk_data()

app.mount("/static", StaticFiles(directory="documents"), name="documents")


app.include_router(user.router)
app.include_router(document.router)


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict[str, str]:
    return {"message": "FastAPI Server!"}