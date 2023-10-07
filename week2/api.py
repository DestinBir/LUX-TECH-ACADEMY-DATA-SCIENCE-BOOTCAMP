from fastapi import FastAPI, Body, HTTPException
import uvicorn
import joblib
import pandas as pd

l_reg = joblib.load('week2/l_reg.pkl')
rf_reg = joblib.load('week2/rf_reg.pkl')

app = FastAPI()


@app.post('/linear')
async def predict(data: Body(...)):

    X = pd.read_json(data)
    
    y_pred = l_reg.predict(X)

    return {'prediction': y_pred}

@app.post('/forest-random')
async def predict(data: Body(...)):
    
    X = pd.read_json(data)
    
    y_pred = rf_reg.predict(X)

    return {'prediction': y_pred}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
