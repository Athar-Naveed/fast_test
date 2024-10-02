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

@app.get("/")
def index():
    return {"message": "Hello, FastAPI!"}

# dynamic path
@app.get("/hello/{username}")
async def hello_world(username:str):
    return {"message": f"Hello,{username}!"}

# Parameters
@app.get("/hi")
async def hi_world(username:str):
    return {"message": f"Hello,{username}!"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", reload=True)