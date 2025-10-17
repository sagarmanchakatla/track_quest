from fastapi import APIRouter, HTTPException
from app.models.runs import StartRunRequest, UpdateRunLocationRequest, EndRunRequest
from datetime import datetime  
from app.services.runs import RunsService
import logging
from uuid import UUID
from app.utils.supabase.supabase_client import supabase

logger = logging.getLogger("RunRoutes")

router = APIRouter(prefix="/running", tags=["Running"])

@router.post("/runs/start")
def start_run(request: StartRunRequest):
    logger.info(f"Received /running/runs/start request from user: {request.user_id}")
    try:
        if not request.user_id:
            logger.error("user_id is required")
            raise HTTPException(status_code=400, detail="user_id is required")
        
        run_data = {
            "user_id": str(request.user_id),
            "started_at": datetime.utcnow().isoformat(),
            "activity_type": request.activity_type,
            "status": "active"
        }
        
        logger.debug(f"Creating run with data: {run_data}")
        response = RunsService.create_run(run_data, request.user_id)
        
        logger.info(f"Run started successfully: {response['id']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{run_id}/location")
def add_run_location(run_id: UUID, request: UpdateRunLocationRequest):
    logger.info(f"Received /running/runs/{run_id}/location request")
    try:
        if not run_id:
            logger.error("run_id is required")
            raise HTTPException(status_code=400, detail="run_id is required")
        
        if not request.location:
            logger.error("location is required")
            raise HTTPException(status_code=400, detail="location is required")
        
        logger.debug(f"Adding location to run: {run_id}")
        response = RunsService.update_run_location(run_id, request.location)
        
        logger.info(f"Location recorded for run: {run_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating run location: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    
@router.post("/runs/{run_id}/end")
def end_run(run_id: UUID, request: EndRunRequest):
    logger.info(f"Received /running/runs/{run_id}/end request")
    try:
        if not run_id:
            logger.error("run_id is required")
            raise HTTPException(status_code=400, detail="run_id is required")
        
        if not request.end_coordinates:
            logger.error("end_coordinates is required")
            raise HTTPException(status_code=400, detail="end_coordinates is required")
        
        end_coords = {
            "latitude": request.end_coordinates.latitude,
            "longitude": request.end_coordinates.longitude
        }
        
        logger.debug(f"Ending run with coordinates: {end_coords}")
        response = RunsService.end_run(run_id, end_coords)
        
        logger.info(f"Run ended successfully: {run_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Add this to your existing FastAPI routes

@router.get("/users/{user_id}/territories")
def get_user_territories(user_id: UUID):
    """Get all territories for a user"""
    try:
        territories = supabase.table("territories")\
            .select("*")\
            .eq("user_id", str(user_id))\
            .order("captured_at", desc=True)\
            .execute()
        
        return territories.data
    except Exception as e:
        logger.error(f"Error fetching territories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))