from faker import Faker
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
from data_generation.schemas import (
    User, UserCreate, UserRole,
    Product, ProductCreate, ProductCategory
)


class RandomDataGenerator:
    def __init__(self, seed: int = None):
        self.fake = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
    
    def generate_user_data(self) -> Dict[str, Any]:
        return {
            "id": self.fake.random_int(min=1, max=10000),
            "username": self.fake.user_name(),
            "email": self.fake.email(),
            "full_name": self.fake.name(),
            "role": random.choice(list(UserRole)),
            "is_active": self.fake.boolean(chance_of_getting_true=85),
            "created_at": self.fake.date_time_between(start_date="-2y", end_date="now")
        }
    
    def generate_user_create_data(self) -> Dict[str, Any]:
        return {
            "username": self.fake.user_name(),
            "email": self.fake.email(),
            "full_name": self.fake.name(),
            "password": self.fake.password(length=12),
            "role": random.choice(list(UserRole))
        }
    
    def generate_user_update_data(self) -> Dict[str, Any]:
        data = {}
        if self.fake.boolean():
            data["username"] = self.fake.user_name()
        if self.fake.boolean():
            data["email"] = self.fake.email()
        if self.fake.boolean():
            data["full_name"] = self.fake.name()
        if self.fake.boolean():
            data["role"] = random.choice(list(UserRole))
        if self.fake.boolean():
            data["is_active"] = self.fake.boolean()
        return data
    
    def generate_product_data(self) -> Dict[str, Any]:
        category = random.choice(list(ProductCategory))
        name = self._generate_product_name(category)
        
        return {
            "id": self.fake.random_int(min=1, max=10000),
            "name": name,
            "description": self.fake.text(max_nb_chars=200),
            "price": round(random.uniform(5.99, 999.99), 2),
            "category": category,
            "stock_quantity": self.fake.random_int(min=0, max=1000),
            "is_available": self.fake.boolean(chance_of_getting_true=90),
            "created_at": self.fake.date_time_between(start_date="-1y", end_date="now"),
            "updated_at": self.fake.date_time_between(start_date="-30d", end_date="now")
        }
    
    def generate_product_create_data(self) -> Dict[str, Any]:
        category = random.choice(list(ProductCategory))
        name = self._generate_product_name(category)
        
        return {
            "name": name,
            "description": self.fake.text(max_nb_chars=200),
            "price": round(random.uniform(5.99, 999.99), 2),
            "category": category,
            "stock_quantity": self.fake.random_int(min=0, max=1000)
        }
    
    def generate_product_update_data(self) -> Dict[str, Any]:
        data = {}
        if self.fake.boolean():
            category = random.choice(list(ProductCategory))
            data["name"] = self._generate_product_name(category)
        if self.fake.boolean():
            data["description"] = self.fake.text(max_nb_chars=200)
        if self.fake.boolean():
            data["price"] = round(random.uniform(5.99, 999.99), 2)
        if self.fake.boolean():
            data["category"] = random.choice(list(ProductCategory))
        if self.fake.boolean():
            data["stock_quantity"] = self.fake.random_int(min=0, max=1000)
        if self.fake.boolean():
            data["is_available"] = self.fake.boolean()
        return data
    
    def _generate_product_name(self, category: ProductCategory) -> str:
        category_names = {
            ProductCategory.ELECTRONICS: [
                "Smartphone", "Laptop", "Tablet", "Headphones", "Smart Watch",
                "Camera", "Speaker", "Monitor", "Keyboard", "Mouse"
            ],
            ProductCategory.CLOTHING: [
                "T-Shirt", "Jeans", "Sweater", "Jacket", "Dress",
                "Shoes", "Sneakers", "Hat", "Scarf", "Gloves"
            ],
            ProductCategory.BOOKS: [
                "Novel", "Biography", "Cookbook", "Manual", "Textbook",
                "Journal", "Comic", "Guide", "Dictionary", "Poetry"
            ],
            ProductCategory.HOME: [
                "Lamp", "Chair", "Table", "Vase", "Pillow",
                "Curtains", "Rug", "Clock", "Picture Frame", "Candle"
            ],
            ProductCategory.SPORTS: [
                "Running Shoes", "Yoga Mat", "Dumbbells", "Basketball",
                "Tennis Racket", "Bicycle", "Helmet", "Water Bottle", "Backpack", "Gloves"
            ]
        }
        
        base_name = random.choice(category_names[category])
        brand = self.fake.company()
        model = self.fake.bothify(text="## ???").upper()
        
        # Sometimes include brand and model
        if self.fake.boolean(chance_of_getting_true=70):
            return f"{brand} {base_name} {model}"
        else:
            return f"{base_name} {model}"
    
    def generate_paginated_response(self, items: List[Dict], page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        total = len(items)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_items = items[start_idx:end_idx]
        
        return {
            "items": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
    
    def generate_error_response(self, status_code: int = 400) -> Dict[str, Any]:
        error_messages = {
            400: ("Bad Request", "The request contains invalid parameters"),
            401: ("Unauthorized", "Authentication is required"),
            403: ("Forbidden", "You don't have permission to access this resource"),
            404: ("Not Found", "The requested resource was not found"),
            422: ("Unprocessable Entity", "The request contains invalid data"),
            500: ("Internal Server Error", "An unexpected error occurred")
        }
        
        error, message = error_messages.get(status_code, ("Unknown Error", "An error occurred"))
        
        return {
            "error": error,
            "message": message,
            "status_code": status_code
        }
    
    def generate_api_request_data(self, endpoint_path: str, method: str) -> Dict[str, Any]:
        if "users" in endpoint_path:
            if method == "POST":
                return self.generate_user_create_data()
            elif method == "PUT":
                return self.generate_user_update_data()
        elif "products" in endpoint_path:
            if method == "POST":
                return self.generate_product_create_data()
            elif method == "PUT":
                return self.generate_product_update_data()
        
        return {}
    
    def generate_api_response_data(self, endpoint_path: str, method: str, status_code: int = 200) -> Dict[str, Any]:
        if status_code >= 400:
            return self.generate_error_response(status_code)
        
        if "users" in endpoint_path:
            if method == "GET" and "{user_id}" not in endpoint_path:
                # List users
                users = [self.generate_user_data() for _ in range(random.randint(1, 20))]
                return self.generate_paginated_response(users)
            else:
                # Single user
                return self.generate_user_data()
        elif "products" in endpoint_path:
            if method == "GET" and "{product_id}" not in endpoint_path:
                # List products
                products = [self.generate_product_data() for _ in range(random.randint(1, 20))]
                return self.generate_paginated_response(products)
            else:
                # Single product
                return self.generate_product_data()
        
        return {}