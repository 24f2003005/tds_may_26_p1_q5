import os
import json
import time
import asyncio
import httpx
from shared import ask_ai_structured

class TelegramProcessor:
    def __init__(self, telegram_api_url: str, base_url: str, log_dir: str = "extra/tgchat", model_id: str = "nvidia/nemotron-3-ultra-550b-a55b:free"):
        self.telegram_api_url = telegram_api_url
        self.base_url = base_url
        self.log_dir = log_dir
        self.model_id = model_id
        self._locks = {}  # Per-chat locks for strict one-by-one execution
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_lock(self, chat_id: str) -> asyncio.Lock:
        if chat_id not in self._locks:
            self._locks[chat_id] = asyncio.Lock()
        return self._locks[chat_id]

    async def _retry_async(self, func, max_retries: int = 3, delay: float = 1.0):
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exc = e
                print(f"[Retry {attempt}/{max_retries}] Failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(delay)
        raise last_exc

    async def process_message(self, chat_id: str, user_id: int, user_text: str) -> dict:
        lock = self._get_lock(chat_id)
        
        # Ensures messages for the same chat run strictly one-by-one
        async with lock:
            log_filename = f"{chat_id}.jsonl"
            log_filepath = os.path.join(self.log_dir, log_filename)
            log_url = f"{self.base_url}/logs/{log_filename}"

            # 1. Load multi-turn history
            history_context = ""
            if os.path.exists(log_filepath):
                with open(log_filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                past_ans = data.get("answer", data.get("response"))
                                history_context += f"User: {data.get('request')}\nAgent: {json.dumps(past_ans)}\n\n"
                            except json.JSONDecodeError:
                                continue

            # 2. Build prompt & schema
            schema = {
                "type": "object",
                "properties": {
                    "answer": {"description": "The exact answer data shaped EXACTLY as requested in the prompt."}
                },
                "required": ["answer"]
            }
            system_prompt = (
                "You are an expert data analyst agent. "
                "The user will ask a data-analysis question and specify the EXACT JSON format they want for the answer. "
                "Output it in the exact structure requested inside the 'answer' key. Do NOT include 'log_url'."
            )
            full_prompt = f"{history_context}Current Question:\n{user_text}"
            start_time = time.perf_counter()

            # 3. LLM call with 3 retries
            try:
                ai_response = await self._retry_async(
                    lambda: ask_ai_structured(
                        question=full_prompt,
                        schema=schema,
                        system_prompt=system_prompt,
                        MODEL_ID=self.model_id,
                        use_openai=False
                    ),
                    max_retries=3
                )
                final_response = {
                    "answer": ai_response.get("answer", ai_response),
                    "log_url": log_url
                }
            except Exception as e:
                final_response = {
                    "answer": {"error": f"Failed after 3 retries: {str(e)}"},
                    "log_url": log_url
                }

            time_taken = time.perf_counter() - start_time

            # 4. Write log
            log_entry = {
                "timestamp": time.time(),
                "user_id": user_id,
                "request": user_text,
                "answer": final_response,
                "time_taken": round(time_taken, 2),
                "model": self.model_id
            }
            with open(log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # 5. Send Telegram reply with 3 retries
            bot_reply_text = json.dumps(final_response)
            
            async def send_tg():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    res = await client.post(
                        f"{self.telegram_api_url}/sendMessage",
                        json={"chat_id": chat_id, "text": bot_reply_text}
                    )
                    res.raise_for_status()

            try:
                await self._retry_async(send_tg, max_retries=3)
            except Exception as err:
                print(f"Telegram delivery failed after retries: {err}")

            return final_response