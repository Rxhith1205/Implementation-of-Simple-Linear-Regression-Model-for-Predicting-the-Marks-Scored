# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ROHITH HARIHARAN M
RegisterNumber: 212223220087
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:


![Screenshot 2025-03-04 094348](https://github.com/user-attachments/assets/93edbadc-8c1e-4b9f-be18-82cd4b3762c3)



![Screenshot 2025-03-04 094354](https://github.com/user-attachments/assets/15ef7856-5d0c-453a-a417-620bf1b920c7)



![Screenshot 2025-03-04 094402](https://github.com/user-attachments/assets/6dc37f77-1389-41eb-9eaa-e5aa89f8beaf)



![Screenshot 2025-03-04 094408](https://github.com/user-attachments/assets/c913cb5d-0448-4437-98ee-73ddbe2dc352)



![Screenshot 2025-03-04 094414](https://github.com/user-attachments/assets/62574521-e1e4-4858-811d-a9cad92b87ff)



![Screenshot 2025-03-04 094422](https://github.com/user-attachments/assets/e1fe49e0-bfbe-403b-bd6e-305c45dd58ba)


![Screenshot 2025-03-04 094428](https://github.com/user-attachments/assets/c074a4fa-305c-488c-bf28-801d19604cdb)



![Screenshot 2025-03-04 094436](https://github.com/user-attachments/assets/9a3a62de-a3d8-4320-b7ac-6b2170bc6201)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
