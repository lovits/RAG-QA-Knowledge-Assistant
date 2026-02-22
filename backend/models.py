from pydantic import BaseModel
from typing import Optional

class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    phone: Optional[str] = None
    avatar: Optional[str] = None
    preferred_output_format: Optional[str] = "markdown"

class UserUpdate(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    avatar: Optional[str] = None
    preferred_output_format: Optional[str] = None
    password: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    avatar: Optional[str] = None
    preferred_output_format: str = "markdown"
