import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline  import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
import os
from sklearn.metrics import mean_squared_error




MODEL_FILE="car_model.pkl"

def build_pipeline(numatr,catatr):
   
    fullp=ColumnTransformer([
        ("num" , RobustScaler() , numatr),
        ("cat" ,OneHotEncoder(drop='first' , handle_unknown="ignore" , sparse_output=False) , catatr)
        
    ])
    return fullp


if not os.path.exists(MODEL_FILE):
    df=pd.read_csv("oldcar.csv")
    x=df.drop('selling_price',axis=1).copy()
    y=df['selling_price'].copy()
    bins = pd.qcut(y, q=10, duplicates="drop")
    xtrain, xtest ,ytrain, ytest=train_test_split(x,y,test_size=0.2, shuffle=True , stratify=bins)
    xtest.to_csv("input.csv",index=False)
    xtest['orig_selling_price']=ytest
    xtest.to_csv("expected_output.csv",index=False)
    num=x.select_dtypes(include=['number']).columns.tolist()
    cat=x.select_dtypes(include=['object']).columns.tolist()
    transformer=build_pipeline(num,cat)
    model_pipeline = Pipeline([
    ('preprocessor', transformer),
    ('regressor', XGBRegressor(
        n_estimators=512,
        learning_rate=0.1,
        min_child_weight= 3,
        max_depth=4,
        random_state=42,
        subsample=0.6995,
        gamma= 1.6475,
        colsample_bytree=0.6859,
        reg_alpha=5.42,
        reg_lambda=9.0
    ))
    ])
    model_pipeline.fit(xtrain,ytrain)
    joblib.dump(model_pipeline, MODEL_FILE )
    print("model is trained Congrats")
else:
   #lets do inference 
    model=joblib.load(MODEL_FILE)

    input_data=pd.read_csv('input.csv')
    predictions= model.predict(input_data)
    predictions=np.round(predictions).astype(int)
    input_data['selling_price']=predictions
    input_data.to_csv("output.csv",index=False)
    print("Predictio is ready result in output.csv!")
    expected_output = pd.read_csv('expected_output.csv')
    mse = mean_squared_error(expected_output['orig_selling_price'], predictions)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
