# Datacamp Python

## Introduction to Python and Intermediate Python

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
s argument means size
col means color
alpha means opacity
```
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])
plt.text(1550, 71, 'India') #additional points
plt.text(5700, 80, 'China')  #additional points
plt.grid(True) # add grid lines
plt.show()
```
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

### remove key-value pair from dictionary. example
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

### dataframe 
Columnaccessbrics[["country", "capital"]]Rowaccess:onlythroughslicingbrics[1:4]loc(label-based)Rowaccessbrics.loc[["RU", "IN", "CH"]]Columnaccessbrics.loc[:, ["country", "capital"]]

### Comparison operators
Before, the operational operators like < and >= worked with Numpy arrays out of the box. Unfortunately, this is not true for the boolean operators and, or, and not.
```
area=np.array([10,15,23])
eg print(area<10)
```
To use these operators with Numpy, you will need np.logical_and(), np.logical_or() and np.logical_not()
eg print(np.logical_and(my_house > 13, 
               your_house < 15))
               
```
# Create medium: observations with cars_per_cap between 100 and 500
cpc=cars['cars_per_cap']
between=np.logical_and(cpc>100, cpc<500)
medium=cars[between]
print(medium)
```

### for loop for lists
```
for height in areas:
   print('height')
```

### using enumerate
```
fam = [1.73, 1.68, 1.71, 1.89]
for index, height in enumerate(fam) :
    print("person " + str(index) + ": " + str(height))
```

### for loop for lists in lists
```
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for y in house:
    print("the " + str(y[0])+ " is " + str(y[1])+" sqm")
```

### for loop for dictionary
```
# Iterate over europe
for country,capital in europe.items():
    print("the capital of "+country+" is "+capital )
```

### every element in 2D np array
A 2D array is built up of multiple 1D arrays. To explicitly iterate over all separate elements of a multi-dimensional array, you'll need this syntax:
```
for x in np.nditer(my_array) :
    ...
```

### Iterating over a Pandas DataFrame 
iterrows() method. Used in a for loop, every observation is iterated over and on every iteration the row label and actual row contents are available:
```
for lab, row in brics.iterrows() :
    ...
    
# example
# Code for loop that adds COUNTRY column
for lab,x in cars.iterrows():
    cars.loc[lab,'COUNTRY']=cars.loc[lab,'country'].upper()
```

### additional: numpy 
ndarray slices are actually views on the same data buffer. If you modify it, it is going to modify the original ndarray as well.
NumPy array slice	
```
>>> a = np.array([1, 2, 5, 7, 8])
>>> a_slice = a[1:5]
>>> a_slice[1] = 1000
>>> a
array([   1,    2, 1000, 7,    8])
# Original array was modified
```

### additional: numpy vs pandas
Though Pandas are built on the top of NumPy and they may appear to be similar, however, they both have unique functionalities and purposes.

Let’s start with NumPy. NumPy is a Python package written in C which is used to perform numerical operations and for processing n-dimensional arrays.

Coming on to Pandas, it is defined as a python package which provides high-performance data manipulation in Python and requires NumPy for operating as it is built on the top of NumPy. Both NumPy and Pandas are open source libraries.

Now majorly the difference between these two lies in their data structure, memory consumption, and usage.

NumPy majorly works with numerical data whereas Pandas works with tabular data.
The data structure in Pandas are Series, Dataframes and Panel whose objects can go upto three. Whereas NumPy has Arrays whose objects can go upto n dimensions.
NumPy consumes less memory as compared to Pandas.
Pandas perform better with the data having 500K rows or more whereas NumPy performances better for 50K rows or less.
Pandas is more widely used in industry than NumPy.

### using 'apply' for function over series
apply function on a column element wise. not for method
pd['new_column']=pd['old_column'].apply(function)
```
for lab, row in brics.iterrows() :
    brics.loc[lab, "name_length"] = len(row["country"])

brics["name_length"] = brics["country"].apply(len)
```

### using 'apply' for methods over series
```
Use .apply(str.upper)
cars["COUNTRY"] = cars['country'].apply(lambda x: x.upper())
```

### additional
Python uses 0-based indexing

### slicing subsets of rows
Slicing using the [] operator selects a set of rows and/or columns from a DataFrame. To slice out a set of rows, you use the following syntax: data[start:stop]

### Slicing Subsets of Rows and Columns 
We can select specific ranges of our data in both the row and column directions using either label or integer-based indexing.

loc is primarily label based indexing. Integers may be used but they are interpreted as a label.
iloc is primarily integer based indexing

### Subsetting Data using Criteria
We can also select a subset of our data using criteria. For example, we can select all rows that have a year value of 2002:
surveys_df[surveys_df.year == 2002]

### random number
```
# Import numpy as np
import numpy as np

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())

# integer
The following call generates the integer 4, 5, 6 or 7 randomly. 8 is not included.
import numpy as np
np.random.randint(4, 8)
```

### random walk example
```
# numpy and matplotlib imported, seed set

# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
# This contains the endpoint of all 500 random walks you've simulated. Store this Numpy array as ends
ends=np_aw_t[-1]
# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()
```

### random walk example 2
```
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
final_tails = []
for x in range(100) :
            tails = [0]
            for x in range(10) :  
                        coin = np.random.randint(0, 2)   
                        tails.append(tails[x] + coin)                                             final_tails.append(tails[-1])
plt.hist(final_tails, bins = 10)
plot.show()
```

# transpose 
transpose random walk to visualize
np.transpose(list)



##  Introduction to importing data in python
### display the contents of your current directory
IPython magic command ! ls 

'with' command is context manager

### Open a file: file
file = open('moby_dick.txt', mode='r')
### Print it
print(file.read())
### Check whether file is closed
print(file.closed)
### Close file
file.close()
### Check whether file is closed
print(file.closed)

readline(): print only the first few lines. When a file called file is open, you can print out the first line by executing file.readline(). If you execute the same command again, the second line will print, and so on.

