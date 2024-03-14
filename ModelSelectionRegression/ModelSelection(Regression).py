import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def multiple_regression(x_train, y_train, x_test, y_test):
    reg=LinearRegression()
    reg.fit(x_train,y_train)
    yp=reg.predict(x_test)
    print(yp)
    np.set_printoptions(precision=2)
    print(np.concatenate((yp.reshape(len(yp),1),y_test.reshape(len(y_test),1)),axis=1))
    print(r2_score(yp,y_test))

def polynomial_regression(x_train, y_train, x_test, y_test):
    p=PolynomialFeatures(degree=4)
    xp=p.fit_transform(x_train)
    reg_p=LinearRegression()
    reg_p.fit(xp,y_train)
    yp=reg_p.predict(p.transform(x_test))
    print(yp)
    np.set_printoptions(precision=2)
    print(np.concatenate((yp.reshape(len(yp),1),y_test.reshape(len(y_test),1)),axis=1))
    print(r2_score(yp,y_test))

def svr(x_train, y_train, x_test, y_test):
    scx=StandardScaler()
    scy=StandardScaler()
    x_train=scx.fit_transform(x_train)
    y_train=scy.fit_transform(y_train)
    reg_s=SVR(kernel='rbf')
    reg_s.fit(x_train,y_train)
    yp=scy.inverse_transform(reg_s.predict(scx.transform(x_test)).reshape(-1,1))
    print(yp)
    np.set_printoptions(precision=2)
    print(np.concatenate((yp.reshape(len(yp),1),y_test.reshape(len(y_test),1)),axis=1))
    print(r2_score(yp,y_test))

def decision_tree_regressor(x_train, y_train, x_test, y_test):
    reg_t=DecisionTreeRegressor(random_state=0)
    reg_t.fit(x_train,y_train)
    yp=reg_t.predict(x_test)
    print(yp)
    np.set_printoptions(precision=2)
    print(np.concatenate((yp.reshape(len(yp),1),y_test.reshape(len(y_test),1)),axis=1))
    print(r2_score(yp,y_test))

def random_forest_regressor(x_train, y_train, x_test, y_test):
    reg_f=RandomForestRegressor(n_estimators=10,random_state=0)
    reg_f.fit(x_train,y_train)
    yp=reg_f.predict(x_test)
    print(yp)
    np.set_printoptions(precision=2)
    print(np.concatenate((yp.reshape(len(yp),1),y_test.reshape(len(y_test),1)),axis=1))
    print(r2_score(yp,y_test))

def main():
    df=pd.read_csv('ModelSelectionRegression\Data.csv')
    x=df.iloc[:, :-1].values 
    y=df.iloc[:, -1:].values 

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    while True:
        print("\n1. Multiple Regression\n2. Polynomial Regression\n3. Support Vector Regression\n4. Decision Tree Regressor\n5. Random Forest Regressor\n6. Exit")
        choice = int(input("Enter your choice: "))
        
        if choice == 1:
            multiple_regression(x_train, y_train, x_test, y_test)
        elif choice == 2:
            polynomial_regression(x_train, y_train, x_test, y_test)
        elif choice == 3:
            svr(x_train, y_train, x_test, y_test)
        elif choice == 4:
            decision_tree_regressor(x_train, y_train, x_test, y_test)
        elif choice == 5:
            random_forest_regressor(x_train, y_train, x_test, y_test)
        elif choice == 6:
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
