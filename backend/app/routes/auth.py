import traceback
from fastapi import APIRouter, Header, HTTPException
from app.models.auth import SignInRequest, SignUpRequest
from app.services.auth_service import AuthService
from app.utils.supabase.supabase_client import supabase
import logging

logger = logging.getLogger("AuthRoutes")

router = APIRouter(prefix="/auth", tags=["Authentication"])

# --- Supabase Auth Routes ---
@router.post("/supabase/signup")
def supabase_sign_up(request: SignUpRequest):
    try:
        auth_data = AuthService.supabase_sign_up(request.email, request.password, request.username)
        user_id = auth_data.user.id
        data = {
            "id": user_id,
            "username": request.username,
            "email": request.email,
            "total_distance": 0,
            "total_territories": 0,
            "rank": None
        }
        return {
            "message": "User registered successfully with Supabase Auth",
            "user": data,
            "session": auth_data.session
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/supabase/signin")
def supabase_sign_in(request: SignInRequest):
    try:
        auth_data = AuthService.supabase_sign_in(request.email, request.password)
        if not auth_data.session:
            raise Exception("Failed to retrieve session")
        user_id = auth_data.user.id
        user_res = supabase.table("users").select("*").eq("id", user_id).single().execute()
        if user_res.error:
            raise Exception(user_res.error.message)
        return {
            "message": "Signed in successfully with Supabase Auth",
            "user": user_res.data,
            "session": auth_data.session
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Database-only Auth Routes ---
@router.post("/signup")
def db_sign_up(request: SignUpRequest):
    try:
        user_data = AuthService.db_sign_up(request.email, request.username, request.password)
        return {
            "message": "User registered successfully in database",
            "user": user_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/signin")
def db_sign_in(request: SignInRequest):
    try:
        user_data = AuthService.db_sign_in(request.email, request.password)
        if not user_data:
            raise Exception("Invalid credentials or user not found")
        return {
            "message": "Signed in successfully with database",
            "user": user_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/validate")
def validate_token(authorization: str = Header(None)):
    """Validate JWT sent in Authorization header"""
    try:
        logger.info("Received /auth/validate request")
        logger.info(f"Authorization header: {authorization}")

        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        user = AuthService.get_user_from_token(authorization)

        logger.info(f"Validation successful for user: {user}")
        return {
            "message": "Token is valid",
            "user": user,
        }

    except HTTPException as e:
        logger.warning(f"Validation failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=401, detail=str(e))