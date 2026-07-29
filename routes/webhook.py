import os
from fastapi import APIRouter, Request, BackgroundTasks
from dotenv import load_dotenv
from services.processor import TelegramProcessor

load_dotenv()

router = APIRouter()

TG_API_KEY = os.getenv("TG_API_KEY")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TG_API_KEY}"
BASE_URL = "https://tdsbot.solana.charity"

processor = TelegramProcessor(telegram_api_url=TELEGRAM_API_URL, base_url=BASE_URL)

@router.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    update = await request.json()

    if "message" in update and "text" in update["message"]:
        chat_id = str(update["message"]["chat"]["id"])
        user_id = update["message"]["from"]["id"]
        user_text = update["message"]["text"]

        # Background task processes sequentially per chat_id without blocking the webhook ACK
        background_tasks.add_task(processor.process_message, chat_id, user_id, user_text)

    return {"status": "ok"}