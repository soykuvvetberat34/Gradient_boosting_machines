from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","Division","NewLeague"]])
y=datas["Salary"]
x_=datas.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=99)


gbm_model=GradientBoostingRegressor()
gbm_params={
    "learning_rate":[0.001,0.1,0.01],
    "max_depth":[3,5,8],
    "n_estimators":[100,200,500],
    "subsample":[1,0.5,0.8]
}
gbm_cv=GridSearchCV(gbm_model,gbm_params,cv=5,n_jobs=-1)
gbm_cv.fit(x_train,y_train)
learning_rate=gbm_cv.best_params_["learning_rate"]
max_depth=gbm_cv.best_params_["max_depth"]
n_estimators=gbm_cv.best_params_["n_estimators"]
subsample=gbm_cv.best_params_["subsample"]
gbm_tuned=GradientBoostingRegressor(subsample=subsample,n_estimators=n_estimators,
                                    max_depth=max_depth,learning_rate=learning_rate)
gbm_tuned.fit(x_train,y_train)

predict=gbm_tuned.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,predict))
print(rmse)





























