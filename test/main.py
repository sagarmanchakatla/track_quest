from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import jwt
import bcrypt
import os
from typing import List, Optional
import math

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fitness_app.db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    total_distance = Column(Float, default=0)
    total_territories = Column(Integer, default=0)
    rank = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Activity(Base):
    __tablename__ = "activities"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    distance = Column(Float)
    steps = Column(Integer)
    route_coordinates = Column(JSON)  # List of [lat, lng] points
    activity_type = Column(String)  # running, walking, cycling
    calories = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Territory(Base):
    __tablename__ = "territories"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    polygon_coordinates = Column(JSON)  # List of [lat, lng] points forming polygon
    area_km2 = Column(Float)
    captured_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ActivityCreate(BaseModel):
    start_time: datetime
    end_time: datetime
    distance: float
    steps: int
    route_coordinates: List[List[float]]
    activity_type: str

class ActivityResponse(BaseModel):
    id: int
    user_id: int
    start_time: datetime
    end_time: datetime
    distance: float
    steps: int
    activity_type: str
    calories: float

class TerritoryCreate(BaseModel):
    polygon_coordinates: List[List[float]]
    activity_id: int

class TerritoryResponse(BaseModel):
    id: int
    user_id: int
    area_km2: float
    captured_at: datetime

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    total_distance: float
    total_territories: int
    rank: int

class LeaderboardEntry(BaseModel):
    rank: int
    username: str
    total_distance: float
    total_territories: int

# Utility Functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

def calculate_distance_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in km"""
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_polygon_area(coordinates: List[List[float]]) -> float:
    """Calculate polygon area using Shoelace formula (in km²)"""
    if len(coordinates) < 3:
        return 0
    
    # Anchor point for coordinate conversion
    lat_anchor = coordinates[0][0]
    lon_anchor = coordinates[0][1]
    
    x_coords = []
    y_coords = []
    
    for lat, lon in coordinates:
        x = (lon - lon_anchor) * (math.pi/180 * 6378.137) * math.cos(math.radians(lat_anchor))
        y = (lat - lat_anchor) * (math.pi/180 * 6378.137)
        x_coords.append(x)
        y_coords.append(y)
    
    area = 0
    for i in range(len(x_coords)):
        j = (i + 1) % len(x_coords)
        area += x_coords[i] * y_coords[j]
        area -= x_coords[j] * y_coords[i]
    
    return abs(area) / 2 / 1000000  # Convert to km²

def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """Check if a point is inside a polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside

def get_convex_hull(points: List[List[float]]) -> List[List[float]]:
    """Get convex hull using Graham scan to simplify self-intersecting paths"""
    if len(points) < 3:
        return points
    
    # Sort points lexicographically
    sorted_points = sorted(points)
    
    def cross(o: List[float], a: List[float], b: List[float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]

def is_closed_loop(coordinates: List[List[float]], threshold_km: float = 0.1) -> bool:
    """Check if route forms a closed loop"""
    if len(coordinates) < 4:
        return False
    
    first_point = coordinates[0]
    last_point = coordinates[-1]
    
    # Calculate distance between first and last point
    distance = calculate_distance_haversine(
        first_point[0], first_point[1],
        last_point[0], last_point[1]
    )
    
    return distance <= threshold_km

def simplify_path(coordinates: List[List[float]], tolerance_km: float = 0.01) -> List[List[float]]:
    """Simplify path using Ramer-Douglas-Peucker algorithm"""
    if len(coordinates) < 3:
        return coordinates
    
    def perpendicular_distance(point: List[float], line_start: List[float], line_end: List[float]) -> float:
        """Calculate perpendicular distance from point to line"""
        if line_start == line_end:
            return calculate_distance_haversine(point[0], point[1], line_start[0], line_start[1])
        
        # Project point onto line
        lon1, lat1 = line_start
        lon2, lat2 = line_end
        lon0, lat0 = point
        
        numerator = abs((lat2 - lat1) * lon0 - (lon2 - lon1) * lat0 + lon2 * lat1 - lat2 * lon1)
        denominator = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        
        if denominator == 0:
            return 0
        
        # Convert to approximate km (rough)
        distance_degrees = numerator / denominator
        return distance_degrees * 111  # 1 degree ≈ 111 km
    
    dmax = 0
    index = 0
    
    for i in range(1, len(coordinates) - 1):
        d = perpendicular_distance(coordinates[i], coordinates[0], coordinates[-1])
        if d > dmax:
            dmax = d
            index = i
    
    if dmax > tolerance_km:
        rec1 = simplify_path(coordinates[:index + 1], tolerance_km)
        rec2 = simplify_path(coordinates[index:], tolerance_km)
        return rec1[:-1] + rec2
    else:
        return [coordinates[0], coordinates[-1]]

# FastAPI App
app = FastAPI(title="INTVL Territory Capture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth Endpoints
@app.post("/auth/register", response_model=Token)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    existing_username = db.query(User).filter(User.username == user_data.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    access_token = create_access_token({"sub": user.email, "user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token({"sub": user.email, "user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

# Activity Endpoints
@app.post("/activities", response_model=ActivityResponse)
def create_activity(activity: ActivityCreate, token: str, db: Session = Depends(get_db)):
    payload = verify_token(token)
    user_id = payload["user_id"]
    
    # Calculate calories (rough estimate: 0.04 * weight * distance, assuming 70kg)
    calories = 0.04 * 70 * activity.distance
    
    db_activity = Activity(
        user_id=user_id,
        start_time=activity.start_time,
        end_time=activity.end_time,
        distance=activity.distance,
        steps=activity.steps,
        route_coordinates=activity.route_coordinates,
        activity_type=activity.activity_type,
        calories=calories
    )
    db.add(db_activity)
    
    # Update user stats
    user = db.query(User).filter(User.id == user_id).first()
    user.total_distance += activity.distance
    
    db.commit()
    db.refresh(db_activity)
    return db_activity

@app.get("/activities/{user_id}", response_model=List[ActivityResponse])
def get_user_activities(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    activities = db.query(Activity).filter(Activity.user_id == user_id).order_by(Activity.created_at.desc()).limit(limit).all()
    return activities

# Territory Endpoints
@app.post("/territories", response_model=TerritoryResponse)
def create_territory(territory: TerritoryCreate, token: str, db: Session = Depends(get_db)):
    payload = verify_token(token)
    user_id = payload["user_id"]
    
    coordinates = territory.polygon_coordinates
    
    # Case 1: Linear route (not closed) - return without territory
    if not is_closed_loop(coordinates):
        raise HTTPException(status_code=400, detail="Route is not a closed loop. Only closed routes create territories.")
    
    # Case 2: Self-intersecting or complex routes - use convex hull
    simplified_coords = simplify_path(coordinates, tolerance_km=0.02)
    
    if len(simplified_coords) < 3:
        raise HTTPException(status_code=400, detail="Insufficient points after simplification")
    
    # Use convex hull for self-intersecting paths to get the outermost boundary
    hull_coords = get_convex_hull(simplified_coords)
    
    if len(hull_coords) < 3:
        raise HTTPException(status_code=400, detail="Could not create valid territory boundary")
    
    # Calculate area using the convex hull
    area = calculate_polygon_area(hull_coords)
    
    # Only create territory if area is significant (at least 0.001 km²)
    if area < 0.001:
        raise HTTPException(status_code=400, detail="Territory area too small")
    
    db_territory = Territory(
        user_id=user_id,
        polygon_coordinates=hull_coords,  # Store simplified hull instead of raw path
        area_km2=area
    )
    db.add(db_territory)
    
    # Update user stats
    user = db.query(User).filter(User.id == user_id).first()
    user.total_territories += 1
    
    # Check for territory overlaps with other users and transfer ownership
    existing_territories = db.query(Territory).filter(Territory.user_id != user_id).all()
    
    for existing in existing_territories:
        # Simple overlap check: if any point of existing territory is inside new territory
        for point in existing.polygon_coordinates:
            if point_in_polygon(point, hull_coords):
                # Transfer territory to current user
                existing.user_id = user_id
                existing.last_updated = datetime.utcnow()
                break
    
    db.commit()
    db.refresh(db_territory)
    return db_territory

@app.get("/territories/{user_id}", response_model=List[TerritoryResponse])
def get_user_territories(user_id: int, db: Session = Depends(get_db)):
    territories = db.query(Territory).filter(Territory.user_id == user_id).all()
    return territories

@app.get("/territories/map/all")
def get_all_territories(db: Session = Depends(get_db)):
    territories = db.query(Territory).all()
    return [
        {
            "id": t.id,
            "user_id": t.user_id,
            "polygon_coordinates": t.polygon_coordinates,
            "area_km2": t.area_km2,
            "captured_at": t.captured_at
        }
        for t in territories
    ]

# Leaderboard Endpoints
@app.get("/leaderboard", response_model=List[LeaderboardEntry])
def get_leaderboard(limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.total_distance.desc()).limit(limit).all()
    return [
        {
            "rank": idx + 1,
            "username": user.username,
            "total_distance": user.total_distance,
            "total_territories": user.total_territories
        }
        for idx, user in enumerate(users)
    ]

@app.get("/leaderboard/territories", response_model=List[LeaderboardEntry])
def get_territory_leaderboard(limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.total_territories.desc()).limit(limit).all()
    return [
        {
            "rank": idx + 1,
            "username": user.username,
            "total_distance": user.total_distance,
            "total_territories": user.total_territories
        }
        for idx, user in enumerate(users)
    ]

# User Endpoints
@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/profile/me")
def get_current_user(token: str, db: Session = Depends(get_db)):
    payload = verify_token(token)
    user_id = payload["user_id"]
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Route Analysis Endpoint
@app.post("/routes/analyze")
def analyze_route(data: dict):
    """Analyze route and determine if it can create a territory"""
    coordinates = data.get("coordinates", [])
    
    if len(coordinates) < 3:
        return {
            "route_type": "too_short",
            "can_create_territory": False,
            "message": "Route has fewer than 3 points"
        }
    
    # Check if closed loop
    is_closed = is_closed_loop(coordinates, threshold_km=0.1)
    
    if not is_closed:
        return {
            "route_type": "linear",
            "can_create_territory": False,
            "message": "Route is not a closed loop. Run in circles or return to start!"
        }
    
    # Try to calculate area
    simplified = simplify_path(coordinates, tolerance_km=0.02)
    hull = get_convex_hull(simplified)
    area = calculate_polygon_area(hull)
    
    if area < 0.001:
        return {
            "route_type": "closed_loop_too_small",
            "can_create_territory": False,
            "area_km2": area,
            "message": f"Territory too small ({area:.6f} km²). Run larger loops!"
        }
    
    return {
        "route_type": "valid_territory",
        "can_create_territory": True,
        "area_km2": round(area, 4),
        "message": f"This route will create a territory of {area:.4f} km²!"
    }

# Health Check
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)