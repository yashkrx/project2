from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# List of allowed origins (can be domains or "*")
origins =["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Who can talk to your API
    allow_credentials=True,           # Whether to send cookies/auth headers
    allow_methods=["*"],               # HTTP methods allowed (GET, POST, etc.)
    allow_headers=["*"],               # HTTP headers allowed
)

@app.get("/")
def read_root():
    return {"message": "CORS works!"}
