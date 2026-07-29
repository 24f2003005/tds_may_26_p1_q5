from fastapi import HTTPException
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import json
import os
import time

load_dotenv() 
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")

# --- Synchronous Clients (For Utility Scripts) ---

def get_openrouter_client_sync() -> OpenAI:
    if not AIPIPE_API_KEY:
        raise HTTPException(status_code=500, detail="AIPIPE_API_KEY not configured")
    return OpenAI(
        base_url="https://aipipe.org/openrouter/v1",
        api_key=AIPIPE_API_KEY,
    )

def get_openai_client_sync() -> OpenAI:
    if not AIPIPE_API_KEY:
        raise HTTPException(status_code=500, detail="AIPIPE_API_KEY not configured")
    return OpenAI(
        base_url="https://aipipe.org/openai/v1",
        api_key=AIPIPE_API_KEY,
    )

# --- Asynchronous Clients (For FastAPI Endpoints) ---

def get_openrouter_client_async() -> AsyncOpenAI:
    if not AIPIPE_API_KEY:
        raise HTTPException(status_code=500, detail="AIPIPE_API_KEY not configured")
    return AsyncOpenAI(
        base_url="https://aipipe.org/openrouter/v1",
        api_key=AIPIPE_API_KEY,
    )

def get_openai_client_async() -> AsyncOpenAI:
    if not AIPIPE_API_KEY:
        raise HTTPException(status_code=500, detail="AIPIPE_API_KEY not configured")
    return AsyncOpenAI(
        base_url="https://aipipe.org/openai/v1",
        api_key=AIPIPE_API_KEY,
    )

# --- Utility Functions ---

def getAllOpenRouterFreeModels():
    client = get_openrouter_client_sync()
    response = client.models.list()
    allModal = []
    data = response.data

    for model in data:
        model_dict = model.model_dump()
        if ':free' in model_dict['id']:
            # Using .get() to prevent KeyError if the provider omits certain fields
            architecture = model_dict.get('architecture') or {}
            allModal.append({
                "id": model_dict.get('id'),
                "name": model_dict.get('name'),    
                "canonical_slug": model_dict.get('canonical_slug'),
                "context_length": model_dict.get('context_length'),
                "modality": architecture.get('modality'),
                "per_request_limits": model_dict.get('per_request_limits'),
                "reasoning": model_dict.get('reasoning'),
            })
            
    print("--- Available Free Models ---", len(allModal))
    
    os.makedirs('extra', exist_ok=True)
    with open('extra/openrouter_free.json', 'w') as f:
        json.dump(allModal, f, indent=4)

def getAllOpenAiModels():
    client = get_openai_client_sync()
    response = client.models.list()
    data = response.data
    
    allModal = []
    print("--- Available OpenAi Models ---", len(data))
    for model in data:
        allModal.append(model.model_dump())

    os.makedirs('extra', exist_ok=True)
    with open('extra/openai_models.json', 'w') as f:
        json.dump(allModal, f, indent=4)


# Models from your list that support the OpenRouter web search tool plugin safely
SUPPORTED_SEARCH_MODELS = {
    "inclusionai/ling-3.0-flash:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-20b:free"
}

# --- Main FastAPI AI Caller ---
async def ask_ai_structured(
    question: str, 
    schema: dict, 
    system_prompt: str = "You are a logical problem solver.",
    MODEL_ID: str = "nvidia/nemotron-3-ultra-550b-a55b:free", 
    use_openai: bool = False
) -> dict:
    start_time = time.perf_counter()
    client = get_openai_client_async() if use_openai else get_openrouter_client_async()
    
    print(f"Sending structured request to model '{MODEL_ID}' (OpenAI API: {use_openai})")
    json_instructions = f"\n\nYou MUST output ONLY valid JSON matching this schema: {json.dumps(schema)}\nDo not include any markdown formatting or text outside the JSON."

    try:
        kwargs = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt + json_instructions},
                {"role": "user", "content": question}
            ],
            "temperature": 0.0,
            "timeout": 60.0 
        }
        
        # Safely add the search tool ONLY if the model is in our approved search-capable list and we aren't using OpenAI directly
        if not use_openai and MODEL_ID in SUPPORTED_SEARCH_MODELS:
            kwargs["tools"] = [{"type": "openrouter:web_search"}]
        
        if use_openai:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True
                }
            }
            
        response = await client.chat.completions.create(**kwargs)
        raw_answer = response.choices[0].message.content
        
        if not raw_answer:
            raise HTTPException(status_code=500, detail="LLM returned an empty response")
            
        start_idx = raw_answer.find('{')
        end_idx = raw_answer.rfind('}')
        if start_idx != -1 and end_idx != -1:
            raw_answer = raw_answer[start_idx:end_idx + 1]
            
        return json.loads(raw_answer)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM did not return valid JSON: {str(e)}\nRaw: {raw_answer}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error during structured LLM call: {exc}")
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Request to '{MODEL_ID}' took {elapsed:.2f} seconds")
        
if __name__ == "__main__":
    getAllOpenRouterFreeModels()
    getAllOpenAiModels()