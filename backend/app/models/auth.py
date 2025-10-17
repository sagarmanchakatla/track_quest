from pydantic import BaseModel

class SignUpRequest(BaseModel):
    email: str
    password: str
    username: str

class SignInRequest(BaseModel):
    email: str
    password: str

class ValidateTokenRequest(BaseModel):
    token: str
