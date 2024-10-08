from fastapi import FastAPI, status
from routers import user
app = FastAPI(debug=True)

app.include_router(user.router)


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict[str, str]:
    return {"message": "FastAPI Server!"}