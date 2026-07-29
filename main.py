import os
from fastapi import FastAPI
from routes import webhook
from routes.items import router as items_router
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="Simple API")

os.makedirs("extra/tgchat", exist_ok=True)
app.mount("/logs", StaticFiles(directory="extra/tgchat"), name="logs")
# Include the router from the routes folder
app.include_router(items_router)
app.include_router(webhook.router)

@app.get("/")
async def root():
    return {"message": "API Root"}