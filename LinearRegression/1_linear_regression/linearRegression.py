import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model

df=pd.read_csv("home_prices.csv")
print(df)

# plot the graph 
# plt.scatter(df.area,df.price)
# plt.xlabel("Area")
# plt.ylabel("Price")
# plt.show()


# now import or make a model linear regression 

reg =linear_model.LinearRegression()

#   y=mx+b  
# y=price ,x=area , m =slope or coefficient and b =intercept of y 


reg.fit(df[['area']],df.price)
print(reg.predict(pd.DataFrame({'area': [3300]})))
print(reg.coef_)
print(reg.intercept_)



plt.scatter(df.area,df.price,color='red',marker='*')
plt.xlabel("Area",fontsize=20)
plt.ylabel("Price",fontsize=20)
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.title("House price prediction")

plt.show()