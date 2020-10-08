# Python - Datacamp
x[2][:2]
x[2] results in a list, that you can subset again by adding additional square brackets.
A list can contain any Python type. But a list itself is also a Python type. That means that a list can also contain a list

A list can contain different types. eg int, float

copy of variable content using list() or using [:]

### list to variable
variable=np.array(variable)

### list vs numpy array
similar but array type is coercion but list can have different types
array 2d is same as list on lists
use of square and curve brackets are same
boolean works on array
calculation for array is different
subsetting and slicing are same

### parameters vs arguments
when defining a function, paramters are written in function header
when call a function, pass arguments into function

### index()
Eg areas.index(20)
for array, just areas[20]

example
this is list on list:
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
this creates a 2D numpy array from list on list
import numpy as np
np_baseball=np.array(baseball)

list can just + to add

mean and median from numpy. use np.mean

### basic plot
import matplotlib.pyplot as plt
### for line
plt.plot(x,y)
### for scatter
plt.scatter(x,y)
### for hist. more for seeing distr
plt.hist(x,bins)
plt.show()
### clean
plt.clf()

### example Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')
plt.grid(True)
plt.show()

### dictionaries
my_dict = {
   "key1":"value1",
   "key2":"value2",
}

to add key-value to dictionaries example
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
europe['italy']='rome'

Dictionaries can contain key:value pairs where the values are again dictionaries.
example
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'norway': { 'capital':'oslo', 'population':5.084 } }
           
select dictionaries elements example
europe['spain']['population']

add new key-value pair to dictionary. example
europe['italy']='rome'

#remove key-value pair from dictionary. example
del(europe['australia'])

The DataFrame is one of Pandas' most important data structures. 
It's a way to store tabular data where you can label the rows and the columns.
Pandas guide: https://pandas.pydata.org/pandas-docs/stable/index.html

dictionaries to dataframe
Each dictionary key is a column label and each value is a list which contains the column elements.
pd.DataFrame()

#pecify row labels(list) for dataframe
variable.index=row_labels

### import csv as dataframe
import pandas as pd
data=pd.read_csv('data.csv')
#specify first column is used as row labels.
data=pd.read_csv('data.csv',index_col=0)

### select column as Pandas Series from dataframe
data['country']

### select column as Pandas DataFrame from dataframe
data[['country']]

### slice rows of dataframe
example: data[0:5] 
Can only use integer indexes of the rows here, not the row labels
OR
loc is label-based, which means that you have to specify rows and columns based on their row and column labels. iloc is integer index based
example select rows:
cars.loc[['RU', 'AUS']]
cars.iloc[[4, 1]]
example select rows and columns:
cars.loc[['IN', 'RU'], 'cars_per_cap']
cars.iloc[[3, 4], 0]

cars.loc[['IN', 'RU'], ['cars_per_cap', 'country']]
cars.iloc[[3, 4], [0, 1]]

### slice columns of dataframe
example:
cars.loc[:, 'country']
cars.iloc[:, 1]

equals sign: <= and >=.
<= is valid syntax, but =< is not.
array can compare

### boolean operators
and, or and not
### boolean operators with array
np.logical_and(), np.logical_or() and np.logical_not(). 
example:n np.logical_and(my_house > 13, your_house < 15)

### if else example
if area > 15 :
    print("big place!")
elif area>10:
    print("medium size, nice!")
else :
    print("pretty small.")
    
### example2
if area>15 :
print("big place!")

### example. drives_right in cars return true
sel = cars[cars['drives_right']]
print(sel)
     cars_per_cap        country  drives_right
US            809  United States          True
RU            200         Russia          True
MOR            70        Morocco          True

observation that i made:
- i think array cannot use .index. dataframe and list can.


