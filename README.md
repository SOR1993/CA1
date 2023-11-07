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

# Statistical Analysis
## general description of the data
df1.describe()

## correlation analysis of the data
df1.corr()

## dataset 2 decriptive statistics
pivot.describe()

## dataset 2correlation analysis of the data
pivot.corr()
## https://seaborn.pydata.org/examples/scatterplot_matrix.html
sns.pairplot(pivot)
plt.grid(True, color = "grey", linewidth = "1", linestyle = "-") ## Source https://www.geeksforgeeks.org/grids-in-matplotlib/
plt.title("Graphs of Various Variables Effecting the Population of Ireland", fontsize=14)
sns.displot(kind='kde', data=pivot, x='Population')
plt.grid(True, color = "grey", linewidth = "1", linestyle = "-") ## Source https://www.geeksforgeeks.org/grids-in-matplotlib/
plt.title("Graph Showing the Kernel Density of the Population of Ireland", fontsize=14)
plt.xlabel("Population", fontsize=12)
plt.ylabel("Density", fontsize=12)

## Poisson Distribution- import poisson
from scipy.stats import poisson

## P(X=65,000) Number of Immigrants equal to 65,000 in a given year
poisson.pmf(k = 65, mu = 65.405405)

## Source: https://www.tutorialspoint.com/how-to-create-a-poisson-probability-mass-function-plot-in-python
lam = 65.405405
# Create an array of x values
x = np.arange(50, 100).tolist()
# Create the Poisson probability mass function
pmf = poisson.pmf(x, lam)
# Create the plot
plt.plot(x, pmf, 'bo', ms=8)
plt.vlines(x, 0, pmf, colors='b', lw=5)
plt.title('Poisson Probability Mass Function- Number of Immigrants')
plt.xlabel('Number of People (thousand)')
plt.ylabel('Probability')
plt.show()

## P(X < 65,000) Probability of the number of immigrants being less than 65,000 in a given year
poisson.cdf(k = 65, mu = 65.405405)
## Source: https://www.tutorialspoint.com/how-to-create-a-poisson-probability-mass-function-plot-in-python
lam = 65.405405
# Create an array of x values
x = np.arange(50, 100).tolist()
# Create the Poisson probability mass function
cdf= poisson.cdf(x, lam)
# Create the plot
plt.plot(x, cdf, 'bo', ms=8)
plt.vlines(x, 0, cdf, colors='b', lw=5)
plt.title('Culmulative Distribution Function- Number of Immigrants')
plt.xlabel('Number of People (thousand)')
plt.ylabel('Probability')
plt.show()

## P(X > 65,000) Probability of the number of immigrants being greater than 65,000 in a given year
poisson.sf(k = 65, mu = 65.405405)
## Source: https://www.tutorialspoint.com/how-to-create-a-poisson-probability-mass-function-plot-in-python
lam = 65.405405
# Create an array of x values
x = np.arange(50, 100).tolist()
# Create the Poisson probability mass function
sf = poisson.sf(x, lam)
# Create the plot
plt.plot(x, sf, 'bo', ms=8)
plt.vlines(x, 0, sf, colors='b', lw=5)
plt.title('Survival Function- Number of Immigrants')
plt.xlabel('Number of People (thousand)')
plt.ylabel('Probability')
plt.show()

## P(X=4,206,000) Population of Ireland being equal to 4,206,000 in a given year
poisson.pmf(k = 4000, mu = 4206.608108)
## Source: https://www.tutorialspoint.com/how-to-create-a-poisson-probability-mass-function-plot-in-python
lam = 4206.608108
# Create an array of x values
x = np.arange(3500, 4500).tolist()
# Create the Poisson probability mass function
pmf = poisson.pmf(x, lam)
# Create the plot
plt.plot(x, pmf, 'g', ms=8)
plt.vlines(x, 0, pmf, colors='g', lw=5)
plt.title('Poisson Probability Mass Function- Population of Ireland')
plt.xlabel('Number of People (thousand)')
plt.ylabel('Probability')
plt.show()

## P(X < 4,206,000) Probability of the population of Ireland being less than 4,206,000 in a given year
poisson.cdf(k = 4206, mu = 4206.608108)
## Source: https://www.tutorialspoint.com/how-to-create-a-poisson-probability-mass-function-plot-in-python
lam = 4206.608108
# Create an array of x values
x = np.arange(3500, 4500).tolist()
# Create the Poisson probability mass function
cdf= poisson.cdf(x, lam)
# Create the plot
plt.plot(x, cdf, 'g', ms=8)
plt.vlines(x, 0, cdf, colors='g', lw=5)
plt.title('Culmulative Distribution Function- Population of Ireland')
plt.xlabel('Number of People (thousand)')
plt.ylabel('Probability')
plt.show()

## P(X > 4,206,000) Probability of the population of Ireland being greater than 4,206,000 in a given year
poisson.sf(k = 4206, mu = 4206.608108)
## Source: https://www.tutorialspoint.com/how-to-create-a-poisson-probability-mass-function-plot-in-python
lam = 4206.608108
# Create an array of x values
x = np.arange(3500, 4500).tolist()
# Create the Poisson probability mass function
sf = poisson.sf(x, lam)
# Create the plot
plt.plot(x, sf, 'g', ms=8)
plt.vlines(x, 0, sf, colors='g', lw=5)
plt.title('Survival Function- Population of Ireland')
plt.xlabel('Number of People (thousand)')
plt.ylabel('Probability')
plt.show()

## Normal Distribution of Population Data
## Testing the distribution for the probability of the number of immigrants being 100,000.

## mu = 65.405405
## sigma = 32.716840
from scipy.stats import norm

## calculate the Z value using the standardisation equation: Z = (x-mu)/std_dev
mean = 65.405405
std_dev = 32.716840
x = 100
z= (x-mean)/std_dev
z

## obtaining the probability 
norm.pdf(100, mean, std_dev)

## plotting using a standard normal distribution Source: https://peterstatistics.com/CrashCourse/Distributions/Normal.html
x = np.arange(-4, 4, 0.001)
leftTail = np.arange(-4, z, 0.001)
plt.plot(x, norm.pdf(x))
plt.fill_between(leftTail, norm.pdf(leftTail), color='red')
plt.title("Graph Showing the Normal Distribution of the Number of Immigrants that Entered Ireland 1987-2023", fontsize=14)
plt.xlabel("Z Value", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.grid(True, color = "grey", linewidth = "1", linestyle = "-")
plt.show()

## Testing the distribution for the probability of the number of immigrants being 35,000.

## mu = 65.405405
## sigma = 32.716840

## calculate the Z value using the standardisation equation: Z = (x-mu)/std_dev
mean = 65.405405
std_dev = 32.716840
x = 35
z= (x-mean)/std_dev
z

## obtaining the probability 
norm.pdf(35, mean, std_dev)

## plotting using a standard normal distribution Source: https://peterstatistics.com/CrashCourse/Distributions/Normal.html
x = np.arange(-4, 4, 0.001)
leftTail = np.arange(-4, z, 0.001)
plt.plot(x, norm.pdf(x))
plt.fill_between(leftTail, norm.pdf(leftTail), color='red')
plt.title("Graph Showing the Normal Distribution of the Number of Immigrants that Entered Ireland 1987-2023", fontsize=14)
plt.xlabel("Z Value", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.grid(True, color = "grey", linewidth = "1", linestyle = "-")
plt.show()

## Testing the distribution for the probability of population of Ireland being 3,500,000 

## mu = 4206.608108
## sigma = 575.151783
from scipy.stats import norm

## calculate the Z value using the standardisation equation: Z = (x-mu)/std_dev
mean = 4206.608108
std_dev = 575.151783
x = 3500
z= (x-mean)/std_dev
z

## obtaining the probability 
norm.cdf(3500, mean, std_dev)

## plotting using a standard normal distribution Source: https://peterstatistics.com/CrashCourse/Distributions/Normal.html
x = np.arange(-4, 4, 0.001)
leftTail = np.arange(-4, z, 0.001)
plt.plot(x, norm.pdf(x))
plt.fill_between(leftTail, norm.pdf(leftTail), color='red')
plt.title("Graph Showing the Normal Distribution of the Population of Ireland", fontsize=14)
plt.xlabel("Z Value", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.grid(True, color = "grey", linewidth = "1", linestyle = "-")
plt.show()

## Testing the distribution for the probability of population of Ireland being 5,500,000 

## mu = 4206.608108
## sigma = 575.151783
from scipy.stats import norm

## calculate the Z value using the standardisation equation: Z = (x-mu)/std_dev
mean = 4206.608108
std_dev = 575.151783
x = 5500
z= (x-mean)/std_dev
z

## obtaining the probability 
norm.cdf(5500, mean, std_dev)

## plotting using a standard normal distribution Source: https://peterstatistics.com/CrashCourse/Distributions/Normal.html
x = np.arange(-4, 4, 0.001)
leftTail = np.arange(-4, z, 0.001)
plt.plot(x, norm.pdf(x))
plt.fill_between(leftTail, norm.pdf(leftTail), color='red')
plt.title("Graph Showing the Normal Distribution of the Population of Ireland", fontsize=14)
plt.xlabel("Z Value", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.grid(True, color = "grey", linewidth = "1", linestyle = "-")
plt.show()

## Machine Learning
## Linear Regression- Multiple Variables
pivot.head()

# Extract the features (X) and target (y)
X = pivot[['Annual births', 'Annual deaths', 'Emigrants', 'Immigrants', 'Natural increase', 'Net migration']].values
y = pivot[['Population change']].values

# Display independent and dependent variables
print(X.shape, y.shape)

# import the libraries for LinearRegression
from sklearn.model_selection import train_test_split

# Call the train_test_split method to split the data and the splitting is 70% for training and 30% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 98)

x.shape, y.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LinearRegression

# Train the LinearRegression mode by using a method fit() function/ method
regression = LinearRegression().fit(x_train, y_train)

print("Training set score: {:.2f}".format(regression.score(x_train, y_train)))
print("Test set score: {:.2f}".format(regression.score(x_test, y_test)))

## Regression Model- Two Variables
# Load the relevant libraries
import pandas
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np

import warnings
warnings.filterwarnings('ignore') # Surpress the warnings

chart1.head()
x = chart1[['Population change']]
y = chart1[['Immigrants']]

sns.scatterplot(x=chart1['Population change'], y=chart1['Immigrants'] )

# create the regression model object and fit the data into it
reg_obj = linear_model.LinearRegression()

# Train the model by calling fit() method
reg_obj.fit(x,y)

# predict the speed at 20:00 hours in the evening and with 750 users online
predicted_population_change = reg_obj.predict([[50]])
print("Predicted Population Change : ")
print(predicted_population_change)

# Display the coefficients
print(reg_obj.coef_)

# Calculate R^2 score for Linear Regression
print("Test set R^2 score: {:.2f}".format(reg_obj.score(x, y)))
# Calculate the mean square error
mean_squared_error = np.mean((predicted_population_change - y)**2)
# Display the mean square error
print("Mean Squared Error on test set", mean_squared_error)


