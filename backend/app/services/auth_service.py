import os
import jwt
import hashlib
import traceback
import logging
from datetime import datetime, timedelta
from fastapi import HTTPException
from app.utils.supabase.supabase_client import supabase

# 🛠️ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AuthService")

# 🔐 JWT Settings
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24


class AuthService:

    # --- 🔑 JWT Creation ---
    @staticmethod
    def create_token(email: str):
        """Create JWT for the given email"""
        logger.info(f"Creating token for: {email}")
        payload = {
            "email": email,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow(),
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        logger.info(f"Generated JWT: {token[:20]}... (truncated)")
        return token

    # --- 🔍 JWT Validation ---
    @staticmethod
    def validate_token(token: str):
        """Decode JWT and return payload"""
        try:
            logger.info(f"Validating token: {token[:25]}... (truncated)")
            if token.startswith("Bearer "):
                token = token.split(" ")[1]
                logger.info("Removed 'Bearer' prefix from token")

            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            logger.info(f"Token payload decoded successfully: {payload}")
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token error: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")

    # --- 👤 Get user from token ---
    @staticmethod
    def get_user_from_token(token: str):
        """Get user data for a valid token"""
        try:
            payload = AuthService.validate_token(token)
            email = payload.get("email")
            if not email:
                raise HTTPException(status_code=401, detail="Token missing email")

            logger.info(f"Fetching user for email: {email}")
            user_res = supabase.table("users").select("*").eq("email", email).single().execute()
            logger.info(f"Supabase response: {user_res}")

            if not user_res.data:
                raise HTTPException(status_code=404, detail="User not found")

            user = user_res.data
            user.pop("password", None)
            logger.info(f"User found: {user}")
            return user

        except Exception as e:
            logger.error(f"Error fetching user from token: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=401, detail=str(e))

    # --- 🧱 Database Sign-Up ---
    @staticmethod
    def db_sign_up(email: str, username: str, password: str):
        try:
            logger.info(f"Sign up attempt — email: {email}, username: {username}")
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            user_data = {
                "email": email,
                "username": username,
                "password": hashed_password,
                "total_distance": 0,
                "total_territories": 0,
                "rank": None,
            }

            logger.info(f"Inserting user into Supabase: {user_data}")
            res = supabase.table("users").insert(user_data).execute()
            logger.info(f"Supabase insert response: {res}")

            if not res.data:
                raise Exception("Failed to insert user into DB")

            user = res.data[0]
            user.pop("password", None)

            token = AuthService.create_token(email)
            logger.info(f"User successfully registered: {user}")

            return {
                "success": True,
                "message": "User registered successfully in database",
                "user": user,
                "token": token,
            }

        except Exception as e:
            logger.error(f"DB Sign Up failed: {e}")
            traceback.print_exc()
            raise Exception(f"DB Sign Up failed: {str(e)}")

    # --- 🔐 Database Sign-In ---
    @staticmethod
    def db_sign_in(email: str, password: str):
        try:
            logger.info(f"Sign in attempt — email: {email}")
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            user_res = supabase.table("users").select("*").eq("email", email).single().execute()
            logger.info(f"Supabase fetch result: {user_res}")

            if not user_res.data:
                raise Exception("User not found")

            if user_res.data["password"] != hashed_password:
                raise Exception("Invalid credentials")

            user = user_res.data
            user.pop("password", None)

            token = AuthService.create_token(email)
            logger.info(f"User authenticated successfully: {user}")

            return {
                "success": True,
                "message": "Signed in successfully with database",
                "user": user,
                "token": token,
            }

        except Exception as e:
            logger.error(f"DB Sign In failed: {e}")
            traceback.print_exc()
            raise Exception(f"DB Sign In failed: {str(e)}")
