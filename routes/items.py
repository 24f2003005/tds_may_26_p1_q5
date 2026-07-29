from fastapi import APIRouter

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/")
async def get_items():
    return {"message": "List of items"}

@router.post("/")
async def create_item():
    return {"message": "Item created"}