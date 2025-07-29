from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"


class ProductCategory(str, Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME = "home"
    SPORTS = "sports"


class User(BaseModel):
    id: int = Field(..., description="Unique user identifier")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    is_active: bool = Field(default=True, description="Whether user is active")
    created_at: datetime = Field(default_factory=datetime.now, description="Account creation timestamp")


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(...)
    full_name: Optional[str] = None
    password: str = Field(..., min_length=8)
    role: UserRole = Field(default=UserRole.USER)


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class Product(BaseModel):
    id: int = Field(..., description="Unique product identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., gt=0, description="Product price")
    category: ProductCategory = Field(..., description="Product category")
    stock_quantity: int = Field(..., ge=0, description="Available stock quantity")
    is_available: bool = Field(default=True, description="Whether product is available")
    created_at: datetime = Field(default_factory=datetime.now, description="Product creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    category: ProductCategory = Field(...)
    stock_quantity: int = Field(..., ge=0)


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    category: Optional[ProductCategory] = None
    stock_quantity: Optional[int] = Field(None, ge=0)
    is_available: Optional[bool] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime


class ProductResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    category: ProductCategory
    stock_quantity: int
    is_available: bool
    created_at: datetime
    updated_at: Optional[datetime]


class PaginatedResponse(BaseModel):
    items: List[dict]
    total: int
    page: int
    per_page: int
    total_pages: int


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int