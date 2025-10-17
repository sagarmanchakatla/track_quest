from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class Coordinate(BaseModel):
    latitude: float
    longitude: float
    
    def to_dict(self):
        return {"latitude": self.latitude, "longitude": self.longitude}    
    
class LocationPoint(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    timestamp: datetime

class StartRunRequest(BaseModel):
    user_id : UUID
    activity_type: str = "running"

class UpdateRunLocationRequest(BaseModel):
    run_id: UUID
    location: LocationPoint

class EndRunRequest(BaseModel):
    run_id: UUID
    end_coordinates: Coordinate