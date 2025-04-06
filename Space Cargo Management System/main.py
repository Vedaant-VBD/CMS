import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import json
from train import SimplifiedCargoRetrievalEnv, discretize_state
from fastapi.responses import HTMLResponse

# Create FastAPI app
app = FastAPI(title="Cargo Management System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("✅ Loaded Q-table successfully")
except Exception as e:
    print(f"❌ Error loading Q-table: {e}")
    q_table = {}

# Global data storage
items_data = []
containers_data = []
placements_data = []
logs_data = []
current_date = datetime.now()

# Model schemas
class Coordinates(BaseModel):
    width: float
    depth: float
    height: float

class Position(BaseModel):
    startCoordinates: Coordinates
    endCoordinates: Coordinates

class Item(BaseModel):
    itemId: str
    name: str
    width: float
    depth: float
    height: float
    priority: int
    expiryDate: str
    usageLimit: int
    preferredZone: str

class Container(BaseModel):
    containerId: str
    zone: str
    width: float
    depth: float
    height: float

class PlacementRequest(BaseModel):
    items: List[Item]
    containers: List[Container]

class PlacementResponse(BaseModel):
    success: bool
    placements: List[Dict[str, Any]]
    rearrangements: List[Dict[str, Any]]

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/placement", response_model=PlacementResponse)
async def placement_recommendations(request: PlacementRequest):
    placements = []
    for idx, item in enumerate(request.items):
        state = discretize_state(item.priority, item.preferredZone, item.expiryDate)
        action = None

        if state in q_table:
            action = int(np.argmax(q_table[state]))
            if action >= len(request.containers):
                action = action % len(request.containers)
        else:
            action = idx % len(request.containers)

        selected_container = request.containers[action]
        placement = {
            "itemId": item.itemId,
            "containerId": selected_container.containerId,
            "position": {
                "startCoordinates": {"width": 0, "depth": 0, "height": 0},
                "endCoordinates": {
                    "width": item.width,
                    "depth": item.depth,
                    "height": item.height
                }
            }
        }
        placements.append(placement)
        items_data.append(item)
        placements_data.append(placement)
        logs_data.append({
            "timestamp": datetime.now().isoformat(),
            "userId": "system",
            "actionType": "placement",
            "itemId": item.itemId,
            "details": {
                "toContainer": selected_container.containerId,
                "reason": "RL-based placement"
            }
        })

    return {
        "success": True,
        "placements": placements,
        "rearrangements": []
    }

@app.get("/api/search")
async def search_item(itemId: Optional[str] = None, itemName: Optional[str] = None, userId: Optional[str] = None):
    if not itemId and not itemName:
        raise HTTPException(status_code=400, detail="Either itemId or itemName must be provided")

    for placement in placements_data:
        if placement["itemId"] == itemId or any(item.name == itemName for item in items_data if item.itemId == placement["itemId"]):
            found_item = next((item for item in items_data if item.itemId == placement["itemId"]), None)
            return {
                "success": True,
                "found": True,
                "item": {
                    "itemId": found_item.itemId,
                    "name": found_item.name,
                    "containerId": placement["containerId"],
                    "zone": next((c.zone for c in containers_data if c.containerId == placement["containerId"]), "Unknown"),
                    "position": placement["position"]
                },
                "retrievalSteps": [
                    {
                        "step": 1,
                        "action": "retrieve",
                        "itemId": found_item.itemId,
                        "itemName": found_item.name
                    }
                ]
            }

    return {"success": True, "found": False, "item": {}, "retrievalSteps": []}

@app.get("/api/waste/identify")
async def identify_waste():
    waste_items = []
    for placement in placements_data:
        item = next((i for i in items_data if i.itemId == placement["itemId"]), None)
        if item and datetime.fromisoformat(item.expiryDate) < current_date:
            waste_items.append({
                "itemId": item.itemId,
                "name": item.name,
                "reason": "Expired",
                "containerId": placement["containerId"],
                "position": placement["position"]
            })

    return {
        "success": True,
        "wasteItems": waste_items
    }

@app.post("/api/simulate/day")
async def simulate_day(simulation_data: Dict[str, Any]):
    global current_date

    if "numOfDays" in simulation_data:
        current_date += timedelta(days=simulation_data["numOfDays"])
    elif "toTimestamp" in simulation_data:
        current_date = datetime.fromisoformat(simulation_data["toTimestamp"].replace("Z", "+00:00"))

    return {
        "success": True,
        "newDate": current_date.isoformat(),
        "changes": {
            "itemsUsed": [],
            "itemsExpired": [],
            "itemsDepletedToday": []
        }
    }

@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("temp_items.csv", "wb") as f:
            f.write(contents)

        df = pd.read_csv("temp_items.csv")
        for _, row in df.iterrows():
            item = Item(**row.to_dict())
            items_data.append(item)

        return {
            "success": True,
            "itemsImported": len(df),
            "errors": []
        }
    except Exception as e:
        return {
            "success": False,
            "itemsImported": 0,
            "errors": [{"row": 0, "message": str(e)}]
        }

@app.get("/api/logs")
async def get_logs(
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    itemId: Optional[str] = None,
    userId: Optional[str] = None,
    actionType: Optional[str] = None
):
    filtered_logs = logs_data

    if startDate:
        start_dt = datetime.fromisoformat(startDate)
        filtered_logs = [log for log in filtered_logs if datetime.fromisoformat(log["timestamp"]) >= start_dt]
    if endDate:
        end_dt = datetime.fromisoformat(endDate)
        filtered_logs = [log for log in filtered_logs if datetime.fromisoformat(log["timestamp"]) <= end_dt]
    if itemId:
        filtered_logs = [log for log in filtered_logs if log.get("itemId") == itemId]
    if userId:
        filtered_logs = [log for log in filtered_logs if log.get("userId") == userId]
    if actionType:
        filtered_logs = [log for log in filtered_logs if log.get("actionType") == actionType]

    return {"logs": filtered_logs}

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=False)
