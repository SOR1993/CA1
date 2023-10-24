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

df1.shape

df.count

df1.count

## checking the data for any duplicate rows 

duplicate_rows_df = df[df.duplicated()] 
print("Number of duplicate rows: ", duplicate_rows_df.shape)

## checking data for the presence of any null values contained in the data

print(df.isnull().sum())

print(df1.isnull().sum())

## using a boxplot to gain insight into the data 

sns.boxplot(x=df['Number of People']) 

plt.title("Boxplot Showing the Population of Ireland 1926-2023", fontsize=14)
plt.xlabel("Population", fontsize=12)

## Analysis of the quartiles and interquartile range of the data

Q1 = df1.quantile(0.25) 
Q3 = df1.quantile(0.75) 
IQR = Q3-Q1 
print(IQR)

df1 = df1[~((df1 < (Q1-1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)] 
df1.shape 

## setting 'Year' as index using .set_index() function. Source:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html

df.set_index('Year')

## plotting data to gain insights into the data set

sns.relplot(data=df, x="Year", y="Number of People", hue="Category", kind="line", height=8)

plt.grid(True, color = "grey", linewidth = "1", linestyle = "-") ## Source https://www.geeksforgeeks.org/grids-in-matplotlib/

plt.title("Population of Ireland 1926-2023", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of People (million)", fontsize=12)

## plotting data to gain insights into the data set

fig, ax = plt.subplots(figsize=(30, 15))
sns.barplot(df, x="Year", y="Number of People", hue="Category", width=.9)

plt.grid(True, color = "grey", linewidth = "1", linestyle = "-")
plt.legend(fontsize="22", loc ="upper left")

plt.title("Population of Ireland 1926-2023", fontsize=36)
plt.xlabel("Year", fontsize=28)
plt.ylabel("Number of People (million)", fontsize=28)
plt.xticks(fontsize=20, rotation = 90) ## https://stackabuse.com/rotate-axis-labels-in-matplotlib/
plt.yticks(fontsize=20)

## Additional Data- Factors Effecting Population

df_new= pd.read_csv("popchange1.csv")
df_new.head(5)

df_new.tail(10)

df_new=df_new.drop(columns=['UNIT','STATISTIC Label'])

df_new.head()

df_new = df_new.rename(columns={"VALUE":"Number of People", "Component":"Category"})
df_new.head(5)

pivot = df_new.pivot(index='Year', columns='Category', values='Number of People')

pivot.head()

pivot.shape

pivot.describe()

## aiming to identify any duplicate rows in the data
duplicate_rows_df = pivot[pivot.duplicated()] 
print("Number of duplicate rows: ", duplicate_rows_df.shape)

## identifiued 36 Null values in the data

print(pivot.isnull().sum())

## as the null values are at the beginning of the data set i.e there was no data available, I decided to drop the values in order to make the remained of the data comparable.

pivot.dropna(axis=0,inplace=True) 

## null values dropped. All categories now have equal number of observations.

print(pivot.isnull().sum())

## used the .corr() function to establish the relationship between the variables which effect population growth, the magnitude of the relationship and the action of the relationship


pivot.corr()

## https://sparkbyexamples.com/pandas/pandas-correlation-of-columns/#:~:text=You%20can%20also%20get%20the,()%20function%20returns%20Pearson's%20correlation.

## dropped the other variables and just conducted a comparison vs. population to quickly identify where the strong/weak relationships existed in the data set

pivot[pivot.columns[1:]].corr()['Population'][:].sort_values(ascending=False).to_frame()

## https://stackoverflow.com/questions/39409866/correlation-heatmap

## used seaborn to plot a heatmap of the correlation analysis to allow me to identify key relationships in the data.

corr = pivot.corr()
sns.heatmap(corr, cmap="Blues", annot=True)
plt.title("Correlations of Factors Effecting Population in Ireland", fontsize=18, loc="center")

df_new.head()

chart1 = df_new.pivot(index='Year', columns='Category', values='Number of People')

chart1.head()

print(chart1.isnull().sum())

chart1.dropna(axis=0,inplace=True) 

chart1.head()

chart1.head()

## created a new dataframe chart1 to compare variables which I identified as having a relationship of interest from the correlation analysis

chart1=chart1.drop(columns=['Annual births', 'Annual deaths', 'Natural increase', 'Population'])

chart1.head()

chart1=chart1.drop(columns=['Population'])

chart1.head()

## creating a line chart to analyse the relationships between variables

sns.relplot(data=chart1, kind="line", height=8, linestyle = "--")

plt.grid(True, color = "grey", linewidth = "1", linestyle = "-")

plt.title("Factors Effecting Population Change in Ireland 1987-2023", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of People (thousand)", fontsize=14)

chart1.corr()

chart2 = df_new.pivot(index='Year', columns='Category', values='Number of People')

chart2.head()

## created a new dataframe chart2 to compare variables which I identified as having a relationship of interest from the correlation analysis


chart2=chart2.drop(columns=['Emigrants', 'Immigrants', 'Population', 'Net migration'])

chart2.head()

chart2=chart2.drop(columns=['Emigrants', 'Immigrants', 'Population', 'Net migration'])

chart2.head()

chart2=chart2.drop(columns=['Emigrants', 'Immigrants', 'Population', 'Net migration'])

chart2.head()

print(chart2.isnull().sum())

chart2.dropna(axis=0,inplace=True) 

print(chart2.isnull().sum())

sns.relplot(data=chart2, kind="line", height=8)

plt.grid(True, color = "grey", linewidth = "1", linestyle = "-")

plt.title("Factors Effecting Population Change in Ireland 1987-2023", fontsize=18)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of People (thousand)", fontsize=12)


