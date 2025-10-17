import os
import json
import logging
from datetime import datetime
from fastapi import HTTPException
from app.utils.supabase.supabase_client import supabase
from app.models.runs import LocationPoint, Coordinate
from uuid import UUID
from app.utils.runs.calculate_distance import (
    calculate_total_distance,
    calculate_territory_type,
    create_polygon_wkt
)

logger = logging.getLogger("RunsService")

def safe_float(value):
    try:
        return float(value) if value is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

def safe_int(value):
    try:
        return int(value) if value is not None else 0
    except (ValueError, TypeError):
        return 0

class RunsService:
    @staticmethod
    def create_run(run_data, user_id):
        """Create a new run session"""
        try:
            logger.info(f"Creating run for user: {user_id}")
            res = supabase.table("runs").insert(run_data).execute()
            
            if res.data and len(res.data) > 0:
                run = res.data[0]
                logger.info(f"Run created successfully: {run['id']}")
                
                return {
                    "id": run["id"],
                    "user_id": user_id,
                    "started_at": run["started_at"],
                    "status": "active"
                }
            else:
                logger.error("No data returned from insert operation")
                raise HTTPException(status_code=400, detail="Failed to create run")
                
        except Exception as e:
            logger.error(f"Error creating run: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def update_run_location(run_id: UUID, location: LocationPoint):
        """Add GPS location point to active run"""
        try:
            logger.info(f"Adding location to run: {run_id}")
            
            # Verify run exists and is active
            run = supabase.table("runs").select("*").eq("id", str(run_id)).execute()

            if not run.data:
                logger.error(f"Run not found: {run_id}")
                raise HTTPException(status_code=404, detail="Run not found")
        
            if run.data[0]["status"] != "active":
                logger.error(f"Run is not active: {run_id}")
                raise HTTPException(status_code=400, detail="Run is not active")
            
            # Insert location point
            location_data = {
                "run_id": str(run_id),
                "latitude": location.latitude,
                "longitude": location.longitude,
                "accuracy": location.accuracy,
                "altitude": location.altitude,
                "speed": location.speed,
                "timestamp": location.timestamp.isoformat(),
                "location": f"POINT({location.longitude} {location.latitude})"
            }
            
            response = supabase.table("run_locations").insert(location_data).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Location recorded for run: {run_id}")
                return {
                    "id": response.data[0]["id"],
                    "run_id": str(run_id),
                    "status": "recorded"
                }
            else:
                raise HTTPException(status_code=400, detail="Failed to record location")
                
        except Exception as e:
            logger.error(f"Error adding location to run: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def end_run(run_id: UUID, end_coordinates: Coordinate):
        """End run session and calculate territory"""
        try:
            logger.info(f"Ending run: {run_id}")
            
            # Fetch all locations for this run
            locations_response = supabase.table("run_locations")\
                .select("*")\
                .eq("run_id", str(run_id))\
                .order("timestamp", desc=False)\
                .execute()
            
            if not locations_response.data or len(locations_response.data) < 2:
                logger.error(f"Not enough location data for run: {run_id}")
                raise HTTPException(status_code=400, detail="Not enough location data to complete run")
            
            # Get run details
            run_response = supabase.table("runs").select("*").eq("id", str(run_id)).execute()
            if not run_response.data:
                logger.error(f"Run not found: {run_id}")
                raise HTTPException(status_code=404, detail="Run not found")
                
            run = run_response.data[0]
            locations = locations_response.data
        
            # Extract coordinates (lon, lat format for GeoJSON)
            coordinates = [(loc["longitude"], loc["latitude"]) for loc in locations]
            
            # Append end coordinates if different from last recorded location
            if end_coordinates and (
                end_coordinates["longitude"] != coordinates[-1][0] or 
                end_coordinates["latitude"] != coordinates[-1][1]
            ):
                coordinates.append((end_coordinates["longitude"], end_coordinates["latitude"]))
                logger.info(f"Added end coordinates to run: {run_id}")
            
            # Calculate distance traveled
            total_distance_km = calculate_total_distance(coordinates)
            logger.info(f"Total distance: {total_distance_km} km")
            
            # Calculate duration
            start_time = datetime.fromisoformat(run["started_at"])
            end_time = datetime.utcnow()
            duration_seconds = int((end_time - start_time).total_seconds())
            logger.info(f"Duration: {duration_seconds} seconds")
            
            # Calculate pace
            pace_min_per_km = (duration_seconds / 60) / total_distance_km if total_distance_km > 0 else 0
            
            # Analyze path and calculate area
            territory_analysis = calculate_territory_type(coordinates)
            logger.info(f"Territory analysis: {territory_analysis}")
            
            territory_id = None
            
            if territory_analysis["is_valid"]:
                area_km2 = territory_analysis["area_km2"]
                calculation_method = territory_analysis["calculation_method"]
                
                # Create territory
                polygon_wkt = create_polygon_wkt(territory_analysis["polygon_coords"])
                
                territory_data = {
                    "user_id": str(run["user_id"]),
                    "polygon_geometry": polygon_wkt,
                    "area_km2": area_km2,
                    "captured_at": start_time.isoformat(),
                    "territory_name": f"Territory {calculation_method.upper()} {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
                }
                
                territory_response = supabase.table("territories").insert(territory_data).execute()
                if territory_response.data:
                    territory_id = territory_response.data[0]["id"]
                    logger.info(f"Territory created: {territory_id}")
            
            # Extract start and end coordinates
            start_coord = coordinates[0]
            end_coord = coordinates[-1]
            
            # Update run with completion data
            update_data = {
                "ended_at": end_time.isoformat(),
                "status": "completed",
                "distance_km": round(total_distance_km, 4),
                "duration_seconds": duration_seconds,
                "pace_min_per_km": round(pace_min_per_km, 2),
                "start_coordinates": json.dumps({
                    "longitude": start_coord[0],
                    "latitude": start_coord[1]
                }),
                "end_coordinates": json.dumps({
                    "longitude": end_coord[0],
                    "latitude": end_coord[1]
                }),
                "start_location": f"POINT({start_coord[0]} {start_coord[1]})",
                "end_location": f"POINT({end_coord[0]} {end_coord[1]})"
            }
            
            if territory_id:
                update_data["territory_id"] = territory_id
            
            supabase.table("runs").update(update_data).eq("id", str(run_id)).execute()
            logger.info(f"Run updated: {run_id}")
            
            # Update user profile
            user_id = run["user_id"]
            user_profile = supabase.table("users").select("*").eq("id", user_id).execute()

            if user_profile.data:
                current = user_profile.data[0]
                area_to_add = territory_analysis["area_km2"] if territory_analysis["is_valid"] else 0
                
                # Debug: Check actual field names in database
                logger.info(f"User profile fields: {list(current.keys())}")
                
                # Use safe conversion functions and match exact database column names
                updated_profile = {
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                # Check which fields actually exist in the database and use correct types
                if "total_distance" in current:
                    current_distance = safe_float(current.get("total_distance", 0))
                    updated_profile["total_distance"] = safe_float(round(current_distance + total_distance_km, 2))
                
                # if "total_distance_km" in current:
                #     current_distance = safe_float(current.get("total_distance_km", 0))
                #     updated_profile["total_distance_km"] = safe_float(round(current_distance + total_distance_km, 2))
                
                if "total_territories" in current:
                    current_territories = safe_int(current.get("total_territories", 0))
                    updated_profile["total_territories"] = safe_int(current_territories + (1 if territory_id else 0))
                
                if "total_territories_count" in current:
                    current_territories = safe_int(current.get("total_territories_count", 0))
                    updated_profile["total_territories"] = safe_int(current_territories + (1 if territory_id else 0))
                
                if "total_area_km2" in current:
                    current_area = safe_float(current.get("total_area_km2", 0))
                    updated_profile["total_area_km2"] = safe_float(round(current_area + area_to_add, 4))
                
                # logger.info(f"updated_info -- ", list(updated_profile.values()))
                # Only update if we have fields to update
                if len(updated_profile) > 1:  # More than just updated_at
                    supabase.table("users").update(updated_profile).eq("id", user_id).execute()
                    logger.info(f"User profile updated: {user_id}")
                else:
                    logger.warning(f"No matching fields found in user profile for user: {user_id}")
                    
                    
            response_data = {
                "run_id": str(run_id),
                "distance_km": round(total_distance_km, 4),
                "duration_seconds": duration_seconds,
                "pace_min_per_km": round(pace_min_per_km, 2),
                "territory_created": territory_id is not None,
                "territory_id": territory_id,
                "area_km2": round(territory_analysis["area_km2"], 6),
                "calculation_method": territory_analysis["calculation_method"],
                "start_longitude": start_coord[0],
                "start_latitude": start_coord[1],
                "end_longitude": end_coord[0],
                "end_latitude": end_coord[1],
                "status": "completed"
            }
            
            logger.info(f"Run completed successfully: {run_id}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error ending run: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))