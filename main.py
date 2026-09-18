# from typing import Annotated
# from fastapi import FastAPI, Query


# app = FastAPI()


# @app.get('/')
# def intro():
#     return {"message": "Hello FastAPI"}


# @app.get("/about")
# def about():
#     return{
#         "name": "Rahul Saraswat",
#         "role": "Data Scientist"
#     }


# @app.get("/store/{store_id}")
# def get_store(store_id: int):
#     return{
#         "store_id": store_id
#     }


# @app.get("/sales")
# def det_sales(store_id: int,
#               days: Annotated[int, Query(ge = 1, le = 30)] = 7):
#     return{
#         "store_id": store_id,
#         "days": days
#     }


from typing import Annotated
from fastapi import FastAPI, Query
from pydantic import BaseModel


app = FastAPI()

class PredictionRequest(BaseModel):
    store_id: Annotated[int, Query(ge = 0)]
    days: Annotated[int, Query(ge= 1, le= 30)]
    promotion: Annotated[int, Query()]


@app.post("/predict")
def create_item(post: PredictionRequest):
    return {
        "store_id": post.store_id,
        "days": post.days,
        "promotion": post.promotion
    }