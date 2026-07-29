import json
import os
import httpx
import asyncio
from dotenv import load_dotenv

from shared import ask_ai_structured

load_dotenv()

TG_API_KEY = os.getenv("TG_API_KEY")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TG_API_KEY}"

TG_WEBHOOK_URL = "https://tdsbot.solana.charity/webhook"

async def set_webhook(webhook_url: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TELEGRAM_API_URL}/setWebhook",
            json={"url": webhook_url}
        )
        return response.json()

async def get_webhook_info():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{TELEGRAM_API_URL}/getWebhookInfo")
        return response.json()

async def delete_webhook():
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{TELEGRAM_API_URL}/deleteWebhook")
        return response.json()

async def test_prompt():
    schema = {
        "type": "object",
        "properties": {
            "state": {"type": "string"},
            "capital": {"type": "string"}
        },
        "required": ["state", "capital"]
    }
    
    try:
        response = await ask_ai_structured(
            question="What is the capital of Assam?",
            schema=schema,
            system_prompt="You are a helpful assistant.",
            MODEL_ID="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            use_openai=False
        )
        print("Test Output:", response)
    except Exception as e:
        print("Test Failed:", e)

async def test_hard_reasoning():
    schema = {
        "type": "object",
        "properties": {
            "step_by_step_analysis": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Detailed logical deduction steps to solve the problem."
            },
            "final_answer_days": {
                "type": "integer"
            }
        },
        "required": ["step_by_step_analysis", "final_answer_days"]
    }
    
    # Complex logic puzzle to test reasoning boundaries (off-by-one trap)
    question = (
        "A snail is at the bottom of a 30-foot well. "
        "Each day, it climbs up 3 feet, but each night, it slips back down 2 feet. "
        "How many days will it take for the snail to escape the well completely?"
    )
    
    print("Testing hard reasoning prompt...\n")
    
    try:
        response = await ask_ai_structured(
            question=question,
            schema=schema,
            system_prompt="You are a strict logician. Break down the edge cases before concluding.",
            MODEL_ID="nvidia/nemotron-3-ultra-550b-a55b:free",
            use_openai=False
        )
        print("Test Output:")
        print(json.dumps(response, indent=2))

    except Exception as e:
        print("Test Failed:", e)

if __name__ == "__main__":
    # print("getting webhook info...")
    # result = asyncio.run(get_webhook_info())
    # print(result)

    # print("testing simple prompt...")
    # asyncio.run(test_prompt())

    asyncio.run(test_hard_reasoning())
    print("Done.")

    ...