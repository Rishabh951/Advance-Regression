# House Price Prediction(Advance-Regression)
A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia. The company is looking at prospective properties to buy to enter the market.
 The company wants to know:  Which variables are significant in predicting the price of a house, and  How well those variables describe the price of a house. 
 Clean the data using various tools
 #Applying the log transformation technique on the SalePrice column to convert into a normal distributed data
df['log_value'] = np.log(df['SalePrice'])
 
** RIDGE AND LASSO REGRESSION**
 # list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
# cross validation

Fitting 5 folds for each of 28 candidates, totalling 140 fits
#Using the best hyper parameter in the ridge Regression
# predict for the training dataset - (y_train_pred = ridge.predict(X_train))
# predict for the test dataset - (y_test_pred = ridge.predict(X_test))
The testing accuracy is:0.8873955927447391
DOUBLING THE HYPERPARAMETER VALUE FOR RIDGE

Same process for lasso regression

Number of predictors selected by double the optimal alpha for lasso are:129

The optimal value of alpha is
•	Ridge – 10
•	Lasso – 0.0004
If we double the value of alpha, we are telling the model to reduce the coefficient of features and in Lasso we will clearly see more features being eliminated. This results in underfitting. After doubling the alpha value following are the important predictor variables.!

Five most important predictor variables are:
•	TotalBsmtSF
•	OverallCond
•	Foundation_PConc,
•	GarageCars, BsmtFinSF1
•	MSZoning_RM
To make sure that model is robust and generalizable:

•	Test it in on both Train and Test dataset and make sure to choose that Hyperparameter for which the accuracy (R square) on both Test and Train is close.
•	Make sure no overfitting is happening. If overfitting is there then we try to add some bias to it (using regularization techniques)
•	Also Try to use lesser number of independent variables to predict the dependent variable. It can be achieved by VIF or applying Regularization (preferably Lasso).




