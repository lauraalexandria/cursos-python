from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import uvicorn

app = FastAPI()

class Passenger(BaseModel): # Entrada dos valores que serão utilizados para predição;
    PassengerId: int
    Pclass: int
    Sex: str
    Age: int

@app.get('/predict')
def (passenger: Passenger):
    data = pd.DataFrame([vars(passenger)])


if __name__ == "__main__":
    uvicorn.run(app = app,
                host = '0.0.0.0',
                port = 8000)