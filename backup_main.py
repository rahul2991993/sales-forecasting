from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title = "My first API",
    description = "Simple FastAPI learning project",
    version = "1.0.0"
)

class StudentRequest(BaseModel):
    name: str
    marks: float = Field(
        ge = 0,
        le = 100
    )

@app.get("/")
def home():
    return {
        "message": "API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }

@app.get("/students/{student_id}")
def get_student(student_id: int):
    return {
        "student_id": student_id
    }

@app.post("/result")
def calculate_result(student: StudentRequest):
    if student.marks >= 40:
        result = "Pass"
    else:
        result = "Fail"

    return {
        "name" : student.name,
        "marks" : student.marks,
        "result" : result
    }

@app.get("/divide")
def divide(a: float, b: float):
    if b == 0:
        raise HTTPException(
            status_code = 400,
            detail = "Division by zero is not allowed"
        )

    return {
        "result" : a / b
    }