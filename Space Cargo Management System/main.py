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

# Global data storage (in a real app, this would be a database)
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

# API Routes
@app.get("/")
def read_root():
    return {"message": "Cargo Management System API", "status": "running"}

# 1. Placement Recommendations API
@app.post("/api/placement", response_model=PlacementResponse)
async def placement_recommendations(request: PlacementRequest):
    # Here you would implement your placement algorithm
    # For now, returning a simple placement example
    return {
        "success": True,
        "placements": [
            {
                "itemId": request.items[0].itemId if request.items else "item1",
                "containerId": request.containers[0].containerId if request.containers else "container1",
                "position": {
                    "startCoordinates": {"width": 0, "depth": 0, "height": 0},
                    "endCoordinates": {"width": 10, "depth": 10, "height": 10}
                }
            }
        ],
        "rearrangements": []
    }

# 2. Item Search and Retrieval API
@app.get("/api/search")
async def search_item(itemId: Optional[str] = None, itemName: Optional[str] = None, userId: Optional[str] = None):
    if not itemId and not itemName:
        raise HTTPException(status_code=400, detail="Either itemId or itemName must be provided")
    
    # Simulate finding an item
    return {
        "success": True,
        "found": True,
        "item": {
            "itemId": itemId or "item1",
            "name": itemName or "Example Item",
            "containerId": "container1",
            "zone": "Zone A",
            "position": {
                "startCoordinates": {"width": 0, "depth": 0, "height": 0},
                "endCoordinates": {"width": 10, "depth": 10, "height": 10}
            }
        },
        "retrievalSteps": [
            {
                "step": 1,
                "action": "retrieve",
                "itemId": itemId or "item1",
                "itemName": itemName or "Example Item"
            }
        ]
    }

# 3. Waste Management API
@app.get("/api/waste/identify")
async def identify_waste():
    # This would use your waste identification ML models
    return {
        "success": True,
        "wasteItems": [
            {
                "itemId": "item2",
                "name": "Expired Item",
                "reason": "Expired",
                "containerId": "container1",
                "position": {
                    "startCoordinates": {"width": 10, "depth": 0, "height": 0},
                    "endCoordinates": {"width": 20, "depth": 10, "height": 10}
                }
            }
        ]
    }

# 4. Time Simulation API
@app.post("/api/simulate/day")
async def simulate_day(simulation_data: Dict[str, Any]):
    global current_date
    
    # Advance the date
    if "numOfDays" in simulation_data:
        current_date += timedelta(days=simulation_data["numOfDays"])
    elif "toTimestamp" in simulation_data:
        current_date = datetime.fromisoformat(simulation_data["toTimestamp"].replace("Z", "+00:00"))
    
    # Process items that would be used
    return {
        "success": True,
        "newDate": current_date.isoformat(),
        "changes": {
            "itemsUsed": [],
            "itemsExpired": [],
            "itemsDepletedToday": []
        }
    }

# 5. Import/Export API
@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Save to a temporary file
        with open("temp_items.csv", "wb") as f:
            f.write(contents)
        
        # Read the CSV file
        df = pd.read_csv("temp_items.csv")
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

# 6. Logging API
@app.get("/api/logs")
async def get_logs(
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    itemId: Optional[str] = None,
    userId: Optional[str] = None,
    actionType: Optional[str] = None
):
    # Filter logs based on parameters
    return {
        "logs": [
            {
                "timestamp": datetime.now().isoformat(),
                "userId": "user1",
                "actionType": "placement",
                "itemId": "item1",
                "details": {
                    "fromContainer": None,
                    "toContainer": "container1",
                    "reason": "Initial placement"
                }
            }
        ]
    }

# Run the server
if __name__ == "__main__":
    # The host='0.0.0.0' is necessary to make the server accessible outside the container
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=False)