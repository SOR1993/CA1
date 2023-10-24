# MSc Data Analytics CA1 
## Data Preparation and Cleaning

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

## read in data contained in a .csv file and display the top 5 rows

df= pd.read_csv("popdata_1926.csv")
df.head(5)

## display the bottom 10 rows.

df.tail(10)

## drop columns which are not relevant 

df=df.drop(columns=['UNIT','STATISTIC Label', 'Single Year of Age'])

df.head()

## rename column headings to allow people observing to have a greater understanding of the data 

df = df.rename(columns={"VALUE":"Number of People", "Sex":"Category"})
df.head(5)

## observing the number of rows and columns

df.shape

## looking closer at the data, particularly the categorical variable 'Category' and identify 3 distinct categories

df.describe(include=object)

df.describe()

## displaying the three distinct categories mentioned earlier using the .unique() function

df["Category"].unique()

## sorting column 'Category' by each category 

df.sort_values(["Category"])

## in order to allow for greater comparison between variables I decided to use the .pivot() function to transpose the data in the 'Category' column into three new columns

df1 = df.pivot(index='Year', columns='Category', values='Number of People').fillna(0)

df1.head(5)

## transposing the data allowed me to gain a greater understanding of the individual components of the 'Category' column

df1.describe()

## used the .corr() function to establish the relationship between the variables, the magnitude of the relationship and the action of the relationship

df1.corr()
