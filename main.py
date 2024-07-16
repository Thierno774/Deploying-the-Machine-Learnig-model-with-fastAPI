## The librairies 

from joblib import load 
from typing import Optional
from fastapi import FastAPI 
from pydantic import BaseModel
from sklearn.datasets import load_iris
import uvicorn

iris = load_iris()
## Load the modele 
loaded_model = load("logreg.joblib")
# create the instance of FastAPI
app = FastAPI()

# Define a class object that helps to realize the request 
class request_body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Define du chemin du point de terminaison (API)

# The get method 
@app.post("/predict/")
## Define the predict method
async def predict(data: request_body)-> dict: 
    # Nouvelles données sur lesquelles ont fait la prédiction 
    new_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    # Predictions
    class_idx = loaded_model.predict(new_data)[0]
    ## the fonction have to return the target value
    return {"class" : iris.target_names[class_idx]}

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    


