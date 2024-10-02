from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv("base_url")
app:FastAPI = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[base_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/hello")
async def hello_world():
    return {"message": "Hello, World!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", reload=True)