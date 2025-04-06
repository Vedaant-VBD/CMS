from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("temp_main:app", host="0.0.0.0", port=8000)
