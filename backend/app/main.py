from fastapi import FastAPI

app = FastAPI(title="SafeRoute API")


@app.get("/health")
def health():
    return {"status": "ok"}
