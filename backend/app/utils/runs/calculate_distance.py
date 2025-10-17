import logging
from math import radians, sin, cos, sqrt, atan2, degrees, asin
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
from typing import Optional, List, Tuple, Dict
import pyproj
from functools import partial
# Add missing import
from shapely.geometry import MultiPoint


logger = logging.getLogger("CalculateDistance")

# Constants for territory calculation
MIN_DISTANCE_KM = 0.1  # Minimum path length to create territory
BUFFER_DISTANCE_M = 15  # Buffer distance in meters (15m = ~street width)
MIN_AREA_KM2 = 0.0001  # Minimum area threshold
CLOSED_LOOP_THRESHOLD_M = 50  # Reduced threshold for closed loops


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometers between two GPS points"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


def calculate_total_distance(coordinates: List[Tuple[float, float]]) -> float:
    """Calculate total distance traveled along the path"""
    total_distance = 0.0
    for i in range(len(coordinates) - 1):
        dist = haversine_distance(
            coordinates[i][1], coordinates[i][0],
            coordinates[i + 1][1], coordinates[i + 1][0]
        )
        total_distance += dist
    return total_distance


def is_closed_loop(coordinates: List[Tuple[float, float]]) -> bool:
    """Check if route forms a closed loop with reduced threshold"""
    if len(coordinates) < 3:
        return False
    
    start_dist = haversine_distance(
        coordinates[0][1], coordinates[0][0],
        coordinates[-1][1], coordinates[-1][0]
    )
    
    # Convert km to meters with reduced threshold
    return start_dist * 1000 <= CLOSED_LOOP_THRESHOLD_M


def create_buffered_path_polygon(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Create a polygon by buffering the entire path.
    This is the main INTVL approach - buffer the run path to create territory.
    """
    if len(coordinates) < 2:
        return []
    
    try:
        # Create LineString from coordinates
        line = LineString(coordinates)
        
        # Define projection: WGS84 to Web Mercator for accurate buffering
        project = partial(
            pyproj.transform,
            pyproj.Proj('EPSG:4326'),  # WGS84
            pyproj.Proj('EPSG:3857')   # Web Mercator
        )
        
        # Transform line to Web Mercator
        line_mercator = transform(project, line)
        
        # Buffer the line (distance in meters)
        buffered_mercator = line_mercator.buffer(BUFFER_DISTANCE_M)
        
        # Transform back to WGS84
        project_back = partial(
            pyproj.transform,
            pyproj.Proj('EPSG:3857'),  # Web Mercator
            pyproj.Proj('EPSG:4326')   # WGS84
        )
        
        buffered_wgs84 = transform(project_back, buffered_mercator)
        
        # Convert to coordinate list
        if hasattr(buffered_wgs84, 'exterior'):
            polygon_coords = list(buffered_wgs84.exterior.coords)
        else:
            # Handle MultiPolygon case
            polygon_coords = list(buffered_wgs84.convex_hull.exterior.coords)
        
        return polygon_coords
        
    except Exception as e:
        logger.error(f"Error creating buffered path: {str(e)}")
        # Fallback: use convex hull
        return create_convex_hull_polygon(coordinates)


def create_convex_hull_polygon(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Create convex hull polygon as fallback"""
    if len(coordinates) < 3:
        return []
    
    try:
        points = [Point(lon, lat) for lon, lat in coordinates]
        multipoint = MultiPoint(points)
        convex_hull = multipoint.convex_hull
        
        if hasattr(convex_hull, 'exterior'):
            return list(convex_hull.exterior.coords)
        else:
            return coordinates + [coordinates[0]]  # Simple closure
            
    except Exception as e:
        logger.error(f"Error creating convex hull: {str(e)}")
        return []


def calculate_polygon_area_km2(coordinates: List[Tuple[float, float]]) -> float:
    """
    Calculate polygon area in square kilometers using geodesic area calculation.
    """
    if len(coordinates) < 3:
        return 0.0
    
    try:
        # Create polygon and calculate geodesic area
        polygon = Polygon(coordinates)
        
        # Use pyproj for accurate area calculation
        geom_area = transform(
            partial(
                pyproj.transform,
                pyproj.Proj('EPSG:4326'),
                pyproj.Proj('EPSG:3857')
            ),
            polygon
        )
        
        area_m2 = geom_area.area
        area_km2 = area_m2 / 1_000_000
        
        return area_km2
        
    except Exception as e:
        logger.error(f"Error calculating polygon area: {str(e)}")
        # Fallback: use shoelace formula
        return calculate_polygon_area_m2(coordinates) / 1_000_000


def calculate_polygon_area_m2(coordinates: List[Tuple[float, float]]) -> float:
    """
    Calculate polygon area in square meters using Shoelace formula.
    """
    if len(coordinates) < 3:
        return 0.0
    
    # Use centroid as origin
    center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
    center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
    
    # Convert to Cartesian coordinates (meters)
    cartesian = []
    for lon, lat in coordinates:
        x = (lon - center_lon) * 111320 * cos(radians(center_lat))
        y = (lat - center_lat) * 110540
        cartesian.append((x, y))
    
    # Shoelace formula
    area = 0.0
    n = len(cartesian)
    for i in range(n):
        x1, y1 = cartesian[i]
        x2, y2 = cartesian[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    
    return abs(area) / 2


def calculate_territory_type(coordinates: List[Tuple[float, float]]) -> Dict:
    """
    Analyze path and calculate territory INTVL-style.
    Main approach: Buffer the entire path to create territory.
    
    Returns:
    - area_km2: Calculated area
    - calculation_method: 'path_buffer' or 'convex_hull' or 'none'
    - polygon_coords: Coordinates of territory polygon
    - is_valid: Whether territory should be created
    """
    # Check if path is long enough
    total_distance = calculate_total_distance(coordinates)
    if total_distance < MIN_DISTANCE_KM:
        logger.info(f"Path too short for territory: {total_distance:.3f} km")
        return {
            "area_km2": 0.0,
            "calculation_method": "none",
            "polygon_coords": [],
            "is_valid": False
        }
    
    # Try buffered path approach first (main INTVL method)
    polygon_coords = create_buffered_path_polygon(coordinates)
    
    if polygon_coords and len(polygon_coords) >= 3:
        area_km2 = calculate_polygon_area_km2(polygon_coords)
        
        result = {
            "calculation_method": "path_buffer",
            "area_km2": round(area_km2, 6),
            "polygon_coords": polygon_coords,
            "is_valid": area_km2 > MIN_AREA_KM2
        }
        logger.info(f"Buffered path territory: {area_km2:.6f} km²")
        return result
    
    # Fallback: convex hull for very irregular paths
    polygon_coords = create_convex_hull_polygon(coordinates)
    if polygon_coords and len(polygon_coords) >= 3:
        area_km2 = calculate_polygon_area_km2(polygon_coords)
        
        result = {
            "calculation_method": "convex_hull",
            "area_km2": round(area_km2, 6),
            "polygon_coords": polygon_coords,
            "is_valid": area_km2 > MIN_AREA_KM2
        }
        logger.info(f"Convex hull territory: {area_km2:.6f} km²")
        return result
    
    # No valid territory
    logger.info("No valid territory created")
    return {
        "area_km2": 0.0,
        "calculation_method": "none",
        "polygon_coords": [],
        "is_valid": False
    }


def create_polygon_wkt(coordinates: List[Tuple[float, float]]) -> str:
    """Create WKT POLYGON for PostGIS"""
    if len(coordinates) < 3:
        raise ValueError("Need at least 3 points for polygon")
    
    # Close polygon if needed
    if coordinates[0] != coordinates[-1]:
        coordinates = coordinates + [coordinates[0]]
    
    # Format: POLYGON((lon lat, lon lat, ...))
    coord_str = ", ".join([f"{lon} {lat}" for lon, lat in coordinates])
    return f"POLYGON(({coord_str}))"


