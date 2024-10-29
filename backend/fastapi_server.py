from fastapi import FastAPI, status
from routers import user, document
from fastapi.staticfiles import StaticFiles
app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="documents"), name="documents")

app.include_router(user.router)
app.include_router(document.router)


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict[str, str]:
    return {"message": "FastAPI Server!"}