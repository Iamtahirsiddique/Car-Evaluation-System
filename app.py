from fastapi import FastAPI, HTTPException
import asyncio
import json

app = FastAPI()

async def load_data_async():
    try:
        # Replace 'data.json' with your actual file path
        with open('data.json', 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

@app.get("/load_data")
async def load_data():
    data = await load_data_async()
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
