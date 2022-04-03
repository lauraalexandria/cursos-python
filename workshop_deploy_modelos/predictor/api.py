import sys
import os

sys.path.insert(1, os.getcwd())

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import uvicorn

from predict import Predict

app = FastAPI()

class Passenger(BaseModel): # Entrada dos valores que serão utilizados para predição;
    PassengerId: int
    Pclass: int
    Sex: str
    Age: int

@app.post('/predict')      # Tem que ser um método post, já que há parâmetros para serem passados
def (passenger: Passenger):
    data = pd.DataFrame([vars(passenger)])
    id = passenger.PassengerId
    del data['PassengerId']
    status, result = predictor.predict(data)
    return JSONResponse({"id": int(id), "score": f'{result*100:.2f}%', "predicao": status})


if __name__ == "__main__":
    uvicorn.run(app = app,
                host = '0.0.0.0',
                port = 8000)