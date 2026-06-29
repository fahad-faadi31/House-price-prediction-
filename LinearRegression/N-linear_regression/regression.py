import numpy as np 
import pandas as pd 
from sklearn import linear_model

df=pd.read_csv("home_prices_multivariable.csv")
print(df)



reg=linear_model.LinearRegression()

reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.predict(pd.DataFrame({'area':[3100],'bedrooms':[5],'age':[40]})))

print(reg.coef_)
print(reg.intercept_)
