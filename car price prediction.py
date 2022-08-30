# dataset is taken from kaggle
#Cars - Purchase Decision Dataset
#https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset


# importing libraries
import pandas as pd    # data manuputation
import numpy as np     # numeric python
import matplotlib.pyplot as plt   # for visualizatin 
import seaborn as sns    # visualizatin 



# reading data set 

car_data=pd.read_csv('car_data.csv')


car_data.info()
car_data.describe
car_data.describe()


car_data.isnull().sum()

#cheking any outliers and analyzing data
plt.scatter(car_data.Age, car_data.AnnualSalary,c='red' ,marker='*')


sns.boxplot(car_data.Age)
sns.boxplot(car_data.AnnualSalary)


x=car_data.iloc[:,[2,3]].values 
y=car_data.iloc[:,4].values



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
st=StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.fit_transform(x_test)

## KNN model and cheching accuracy with concusion metrix
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='euclidean',p='2')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)   # got 93% accuracy


## applying linear regression and checking accuracy witn MSE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
linearr=LinearRegression()
linearr.fit(x_train,y_train)
y_pred=linearr.predict(x_test)
mean_squared_error(y_test, y_pred)*100 ## got 11.7% error


## by applying this two ml algorithm we got better result in KNN
