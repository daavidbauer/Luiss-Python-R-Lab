# Luiss-Python-R-Lab
This holds the code of the final project for my Python & R for Data science. 
The goal of this project was to take a dataset containing information about video games (e.g., number of plays, genres etc.) to subject it to a OSEMN pipeline. 
The focus was put on obtaining the data (O), scrubbing and cleaning the data (S) and performing an explorative data analysis (EDA) (E), so the first three steps of the pipeline. 
The modeling (M) and data interpretation (N) were also considered and out of curiosity performed to gain the experience of a full OSEMN pipeline. 
# Cleaning
After importing the data, descriptive statistics are preformed to get some overview on null values, column data types and general layout of the data frame.
Second the very few null values and duplicate rows are removed. 
Next data types and representation of numeric columns represented as strings are properly transformed. 
Titles are cleaned and adjusted and more complicated columns with multiple categorical values per row (Genres) are simplified to allow later utlization as dummy variables.
Dates are split into different columns for year month and season to increase number of potential predictors for the later modeling phase.
Out of curiosity a bag of words k means clustering is performed to get attempt the semantic categorization of otherwise to complicated columns containing entire texts (only in Python)
# EDA
Afterwords various graphs, plots, charts, Pearson correlation matrix and more to get some insight on the data.
# Modeling and Interpretation
Large numeric variables are scaled to stabalize the performance of the later modeling.
For the simple regression performed, categorical variables are split into dummies, the data split into training and test data, the model trained, and the prediction accuracy tested. 
Due to the decreased focus on this last part of the OSEMN pipeline, the performance of the model is very poor (also doe to very low correlation among the variables) and consequently there was
also no attempt made to improve it. 
