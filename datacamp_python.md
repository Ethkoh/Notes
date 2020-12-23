# Datacamp Python

Cheat Sheets: https://www.datacamp.com/community/data-science-cheatsheets

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

### transpose 
transpose random walk to visualize
np.transpose(list)

## Toolbox Part 1
print() is type NoneType

### scopes searched (LEGB rule):
in a function, python will first look for local scope. if no local scope defined, then will look for global scope
1. local
2. enclosing functions
3. global
4. builtin/ predefined

print list of all the names in module/package
dir(module)


### alter global in function
```
def ...():
    global var
    var = ...
    return var
```

### alter name in enclosing scope in nested function
```
def ...():
    var=...
    def ...():
    nonlocal var
    var = ...
    return var
```
```
# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word = word * 2
    
    # Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        # Use echo_word in nonlocal scope
        nonlocal echo_word
        
        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word+"!!!"
    
    # Call function shout()
    shout()
    
    # Print echo_word
    print(echo_word)

# Call function echo_shout() with argument 'hello'
echo_shout('hello')
```

### nested function
example:
```
def mod2plus5(x1, x2, x3):
    def inner(x):
    return x % 2 + 5
    
    return (inner(x1), inner(x2), inner(x3))
    
print(mod2plus5(1, 2, 3))(6, 5, 6)
```
another example:
```
def raise_val(n):
    def inner(x):
        raised = x ** n
        return raised
    return inner

square = raise_val(2)
cube = raise_val(3)
print(square(2), cube(4))
```

### flexible arguments
dont need to specify specific number of arguments

single * example:
turns all the arguments passed to the function into a tuple called args
```
def add_all(*args):
# Initialize sum
sum_all = 0
# Accumulate the sum
for num in args:
sum_all += num
return sum_all
```

double ** example:
```
def print_all(**kwargs):
# Print out the key-value pairs
for key, value in kwargs.items():
print(key + \": \" + value)
print_all(name="dumbledore", job="headmaster")
```
double ** example:
```
# Define report_status
def report_status(**kwargs):

    # Iterate the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name='luke',affiliation="jedi",status="missing")

# Second call to report_status()
report_status(name='anakin', affiliation='sith lord', status='deceased')
```

### map(func,seq)
map() applies function to all elements in the sequence

example:
```
nums = [48, 6, 9, 21, 1]
square_all = map(lambda num: num ** 2, nums)
print(square_all)
<map object at 0x103e065c0>
print(list(square_all))
[2304, 36, 81, 441, 1]
```

### filter and lambda
```
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda x: len(x)>6, fellowship)

# Convert result to a list: result_list
result_list=list(result)

# Print result_list
print(result_list)
```

### Error handling with error raising
example:
```
def sqrt(x):
"""Returns the square root of a number."""
    if x < 0:
        raise ValueError('x must be non-negative')
    try:
        return x ** 0.5
    except TypeError:
        print('x must be an int or float')
```

example raise error if column name not in column:
```
def ...
....

  # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' + col_name + ' column.')

.....
```

### Error handling with try-except
example:
```
def sqrt(x):
"""Returns the square root of a number."""
try:
return x ** 0.5
except:
print('x must be an int or float')
```

## Toolbox Part 2


### iter()
applying to an iterable (eg list) creates an iterator (iter(), object with associate next method)
use next() function to retrieve the values one by one from the iterator object

```
word = 'Da'
it = iter(word)
next(it)
'D'
```
```
# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
```

### iterating at once with *
```
word =
'Data'
it = iter(word)
print(*it)
D a t a
```

### iterating over file connections
```
file = open('file.txt')
it = iter(file)
print(next(it))
This is the first line.
print(next(it))
This is the second line.
```

range() doesn't actually create the list; instead, it creates a range object with an iterator that produces the values until it reaches the limit

fun fact: The value 10100 is actually what's called a Googol which is a 1 followed by a hundred 0s. That's a huge number!

### enumerate()
for list to unpack index and values
```
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
e = enumerate(avengers)
print(type(e))
<class 'enumerate'>
e_list = list(e)
print(e_list)
[(0, 'hawkeye'), (1, 'iron man'), (2, 'thor'), (3, 'quicksilver')]
```

```
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
for index, value in enumerate(avengers):
print(index, value)
```

### zip()
accept arbitray number of iterables and returns iterator of tuples

```
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(type(z))
<class 'zip'>
z_list = list(z)
print(z_list)
[('hawkeye', 'barton'), ('iron man', 'stark'),
('thor', 'odinson'), ('quicksilver', 'maximoff')]
```

### create dictionary from zip lists
```
# Zip lists: zipped_lists
zipped_lists = zip(feature_names,row_vals)

# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)

# Print the dictionary
print(rs_dict)
```

### to create list of dictionary

```
# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names,sublist) for sublist in row_lists]

# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[0])
print(list_of_dicts[1])
```

### list of dics to Dataframe
```
# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)
```

### unpack zip()
```
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
for z1, z2 in zip(avengers, names):
print(z1, z2)
```

### unpack zip() with *
There is no unzip function for doing the reverse of what zip() does. We can, however, reverse what has been zipped together by using zip() with a little help from *! * unpacks an iterable such as a list or a tuple into positional arguments in a function call.

once "unzip" with *, zip becomes "empty"
```
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(*z)
('hawkeye', 'barton') ('iron man', 'stark')
('thor', 'odinson') ('quicksilver', 'maximoff')
```

```
# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants,powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)
```

### create list of tuples from list of strings
```
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list((enumerate(mutants)))
```

### additional
parameter is what is declared in the function, while an argument is what is passed through when calling the function

### iterating over data: loading data in chunks
Sometimes, the data we have to process reaches a size that is too much for a computer's memory to handle. A solution to this is to process an entire data source chunk by chunk, instead of a single go all at once.

```
import pandas as pd
result = []
for chunk in pd.read_csv('data.csv', chunksize=1000):
result.append(sum(chunk['x']))
total = sum(result)
print(total)
4252532
```
alternative:
```
import pandas as pd
total = 0
for chunk in pd.read_csv('data.csv', chunksize=1000):
total += sum(chunk['x'])
print(total)
4252532
```
example function to iterate over file:
```
# Define count_entries()
def count_entries(csv_file,c_size,colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file,chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

```

### list comprehension
create lists from other lists,dataframe,columns etc.
more efficient than for loop

iterate over any iterables not just lists. 

### nested loops for list comphrehension
```
pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6,
print(pairs_2)
[(0, 6), (0, 7), (1, 6), (1, 7)]
```

### matrices
One of the ways in which lists can be used are in representing multi-dimension objects such as matrices. Matrices can be represented as a list of lists in Python.

```
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]
```


### if-else in list comprehension
```
# Create list comprehension: new_fellowship
new_fellowship = [member if len(member)>6 else '' for member in fellowship ]
```

### generator object vs list comprehension
generators returns generator object. dont store elements in memory
list comphresion returns a list
both are iterable
things that list comprehension can do can be done on generators also.
 Generators allow users to lazily evaluate data. This concept of lazy evaluation is useful when you have to deal with very large datasets because it lets you generate values in an efficient manner by yielding only chunks of data at a time instead of the whole thing at once.

```
# Create generator object: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))


# Print the rest of the values
for value in result:
    print(value)
```

```
result = (num for num in range(6))
for num in result:
print(num)
```

```
# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)

```

### generator function: yield() 
generator functions produces generators objects when called.
define like regular function
yields sequence of values
example:
```
def num_sequence(n):
"""Generate values from 0 to n."""
i = 0
while i < n:
yield i
i += 1
```

```
# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)
        

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)
```

### file connection: context manager
Note that when you open a connection to a file, the resulting file object is already a generator! So out in the wild, you won't have to explicitly create generator objects in cases such as this.
```
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(0, 1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)
```

### reading data in chunks
```
# Import the pandas package
import pandas as pd

# Initialize reader object: df_reader
df_reader = pd.read_csv('ind_pop.csv', chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))
```

### append dataframe to dataframe
A useful shortcut to concat() are the append() instance methods on Series and DataFrame. These methods actually predated concat. They concatenate along axis=0, namely the index.

https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

### append dataframe,subsetting, chunksize, zip, list comprehension,scatter plot
```
# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Print pops_list
        print(pops_list)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop(fn,'CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn,'ARB')
```

##  Introduction to importing data in python

### display the contents of your current directory
starting a line with ! gives you complete system shell access. 
command: ls or ! ls

### Open a file: file
file = open('moby_dick.txt', mode='r')
r means read
w means write
### Print it
print(file.read())
### Check whether file is closed
print(file.closed)
### Close file
file.close()
### Check whether file is closed
print(file.closed)

### Context manager with
create context by executing commands with file open.
once out of this contet, file no longer open.
means no need close file.
bind a variable 'file' by using a context manager construct.
```
with open('huck_finn.txt', 'r') as file:
print(file.read())
```


### print lines only
readline(): print only the first few lines. When a file called file is open, you can print out the first line by executing file.readline(). If you execute the same command again, the second line will print, and so on.

the variable file will be bound to open('huck_finn.txt'); thus, to print the file to the shell, all the code you need to execute is:
```
with open('huck_finn.txt') as file:
    print(file.readline())
```

### Flat files
- Text files containing records
- table data
- Record: row of fields or attributes
eg .txt, .csv

.xlsx is not a flat because it is a spreadsheet consisting of many sheets, not a single table.

### Importing flat files using NumPy
'''
filename = 'MNIST_header.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[0, 2])
print(data)
'''
There are a number of arguments that np.loadtxt() takes that you'll find useful:
- delimiter changes the delimiter that loadtxt() is expecting.You can use ',' for comma-delimited.
You can use '\t' for tab-delimited.
- skiprows allows you to specify how many rows (not indices) you wish to skip
- usecols takes a list of the indices of the columns you wish to keep.

### Customizing your NumPy import
load as string type instead of numerical
```
data = np.loadtxt(filename, delimiter=',', dtype=str)
```
can load as 'float' also
however, loadtxt will break down if got mixed datatypes
so need to use np.genfromtxt()

### mixed datatypes in numpy (np.genfromtxt(), np.recfromcsv())
If we pass dtype=None to it, it will figure out what types each column should be.
```
data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
```
Here, the first argument is the filename, the second specifies the delimiter , and the third argument names tells us there is a header. Because the data are of different types, data is an object called a structured array. Because numpy arrays have to contain elements that are all the same type, the structured array solves this by being a 1D array, where each element of the array is a row of the flat file imported. You can test this by checking out the array's shape in the shell by executing np.shape(data).
Accessing rows and columns of structured arrays is super-intuitive: to get the ith row, merely execute data[i] and to get the column with name 'Fare', execute data['Fare'].

There is also another function np.recfromcsv() that behaves similarly to np.genfromtxt(), except that its default dtype is None
```
data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
```
same as 
```
data = np.recfromcsv('titanic.csv', delimiter=',', names=True)
```

### numpy array from dataframe
```
# Build a numpy array from the DataFrame: data_array
data_array=data.values
```

### import using pandas 
- sep - the pandas version of delim
- comment takes characters that comments occur after in the file
- na_values takes a list of strings to recognize as NA/NaN, in this case the string 'Nothing'.

example:
 import a slightly corrupted copy of the Titanic dataset titanic_corrupt.txt, which
- contains comments after the character '#'
- is tab-delimited.
```
# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')
```

new type of dataframe introduced in 2016 is feather.

### plot histogram
```
# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()
```

### picked files
- File type native to Python
- Motivation: many datatypes for which it isn’t obvious how to eg dictonaries (usually stored as json)
- store them
- Pickled files are serialized
- Serialize = convert object to bytestream

easier to import to python

```
import pickle
with open('pickled_fruit.pkl', 'rb') as file:
data = pickle.load(file)
print(data)
```
rb means read and binary only. file that has been pickled. computer readable not human readable.

### library os in python
natively in Python using the library os, which consists of miscellaneous operating system interfaces.

The first line of the following code imports the library os, the second line stores the name of the current directory in a string called wd and the third outputs the contents of the directory in a list to the shell.
```
import os
wd = os.getcwd()
os.listdir(wd)
```

### Importing Excel spreadsheets
```
import pandas as pd
file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
# print sheet names
print(data.sheet_names)
['1960-1966', '1967-1974', '1975-2011']

# to load sheet via name or index
# All these arguments can be assigned to lists containing the specific row numbers, strings and column numbers, as appropriate.
df1 = data.parse('1960-1966') # sheet name, as a string
df2 = data.parse(0) # sheet index, as a float
```

```
# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = pd.ExcelFile('battledeath.xlsx')

# Load spreadsheet: xls
xls = pd.ExcelFile(file)

# Print sheet names
print(xls.sheet_names)

# Load a sheet into a DataFrame by name: df1
df1 = xls.parse('2004')

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2=xls.parse(0)

# Print the head of the DataFrame df2
print(df2.head())
```

### Customizing spreadsheet import
```
# Parse the first sheet by index. In doing so, skip the first row of data and name the columns 'Country' and 'AAM due to War (2002)' using the argument names. The values passed to skiprows and names all need to be of type list.

# Parse the second sheet by index. In doing so, parse only the first column with the usecols parameter, skip the first row and rename the column 'Country'. The argument passed to usecols also needs to be of type list

# Parse the first sheet and rename the columns: df1
df1 = xls.parse(0, skiprows=1, names=['Country','AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xls.parse(1, usecols=[0], skiprows=1, names=['Country'])

# Print the head of the DataFrame df2
print(df2.head())

```


### SAS and Stata files
SAS: Statistical Analysis System
Stata: “Statistics” + “data”
SAS: business analytics and biostatistics
Stata: academic social sciences research

### Importing SAS files
```
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sas7bdat') as file:
df_sas = file.to_data_frame()
```


### Importing Stata files
```
import pandas as pd
data = pd.read_stata('urbanpop.dta')
```

### hdf5 file
- Hierarchical Data Format version 5
- Standard for storing large quantities of numerical data
- Datasets can be hundreds of gigabytes or terabytes
- HDF5 can scale to exabytes
- becoming popular way to store large data

### Importing HDF5 files
```
import h5py
filename = 'H-H1_LOSC_4_V1-815411200-4096.hdf5'
data = h5py.File(filename, 'r') # 'r' is to read
print(type(data))
<class 'h5py._hl.files.File'>
```
example:
```
# Import packages
import numpy as np
import h5py

# Assign filename: file
file='LIGO_data.hdf5'

# Load file: data
data = h5py.File(file, 'r')

# Print the datatype of the loaded file
print(type(data))

# Print the keys of the file
for key in data.keys():
    print(key)

```

### The structure of HDF5 files
```
for key in data.keys():
print(key)
meta
quality
strain
print(type(data['meta']))
<class 'h5py._hl.group.Group'>
```

example:
```
for key in data['meta'].keys():
print(key)

Description
DescriptionURL
Detector
Duration
GPSstart
Observatory
Type
UTCstart

print(data['meta']['Description'].value, data['meta']['Detector'].value)
b'Strain data time series from LIGO' b'H1'
```

### MATLAB
- “Matrix Laboratory”
- Industry standard in engineering and science
- Data saved as .mat  les
- powerful in linear algebra and marix capability
- scipy.io.loadmat() - read .mat files
- scipy.io.savemat() - write .mat files

### Importing a .mat file
```
import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))
<class 'dict'>

print(type(mat['x']))
<class 'numpy.ndarray'>
```
keys = MATLAB variable names
values = objects assigned to variables

### What is a relational database?
- tables are like dataframes
- each rows is an instance of the entity
- each column represents an attribute of each instance
- must have key for each table
- tables are linked
- eg postgreSQL, SQLite, MySQL
- to access many RDMS can use SQLAlchemy package

### Creating a database engine 
to communicate queries with database

sqlite:///name.sqlite is known as communication string
```
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Northwind.sqlite')

# geting table names
table_names = engine.table_names()
print(table_names)
```

### Workflow of SQL querying
1. Import packages and functions
2. Create the database engine
3. Connect to the engine
4. Query the database
5. Save query results to a DataFrame
6. Close the connection

```
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine connection: con
print(engine.table_names())
con=engine.connect()

# creates a SQLAlchemy result object assigned to rs
# Perform query: rs
rs = con.execute("SELECT * FROM Album")

# Save results of the query to DataFrame: df
# fetchall fetches all rows
df = pd.DataFrame(rs.fetchall())

# Close connection
con.close()

# Print head of DataFrame df
print(df.head())

```

Using the context manager
and if only select a few columns
```
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title FROM Employee')
    df = pd.DataFrame(rs.fetchmany(size=3))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print(len(df))

# Print the head of the DataFrame df
print(df.head())
```

using pandas
```
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine=create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT OrderID, CompanyName FROM Orders
INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID", engine)

# Print head of DataFrame
print(df.head())

```
to confirm that both methods yield the same result equals(df))


## Supervised learning with scikit learn/sklearn

other libraries are tensorflow, keras

### what is machine learning
Giving computers the ability to learn to make decisions from data without being explicitly programmed

### what is supervised learning
- Predictor variables/features and a target variable
- Aim: Predict the target variable, given the predictor variables
- Classiffcation: Target variable consists of categories
- Regression: Target variable is continuous
- Automate time-consuming or expensive manual tasks
Example: Doctor’s diagnosis
- Make predictions about the future
Example: Will a customer click on an ad or not?
- Need labeled data
    - Historical data with labels
    - Experiments to get labeled data
    - Crowd-sourcing labeled data

### what is unsupervised learning
Uncovering hidden patterns from unlabeled data

### Naming conventions
- Features = predictor variables = independent variables
- Target variable = dependent variable = response variable

### The Iris dataset in scikit-learn
```
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris = datasets.load_iris()
#  type Bunch, which are dictionary-like objects with keys and values
# can access the keys of these Bunch objects in two different ways: By using the . notation, as in digits.images, or the [] notation, as in digits['images'].
type(iris)
sklearn.datasets.base.Bunch
print(iris.keys())
dict_keys(['data', 'target_names', 'DESCR', 'feature_names', 'target'])
type(iris.data), type(iris.target)
(numpy.ndarray, numpy.ndarray)
iris.data.shape
(150, 4)
iris.target_names
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8, 8],
s=150, marker = 'D')
```

### sns.countplot()
```
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
```
In sns.countplot(), we specify the x-axis data to be 'education', and hue to be 'party'. Recall that 'party' is also our target variable. So the resulting plot shows the difference in voting behavior between the two parties for the 'education' bill, with each party colored differently. We manually specified the color to be 'RdBu', as the Republican party has been traditionally associated with red, and the Democratic party with blue.


### k-Nearest Neighbors
- Basic idea: Predict the label of a data point by
Looking at the ‘k’ closest labeled data points
- Taking a majority vote
- a classiifier
- named the feature array X and response variable y: This is in accordance with the common scikit-learn practice.
- .score() is accuracy

Larger k = smoother decision boundary = less complex model
Smaller k = more complex model = can lead to overfitting
```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
KNeighborsClassifier(algorithm='auto', leaf_size=30,
metric='minkowski',metric_params=None, n_jobs=1,
n_neighbors=6, p=2,weights='uniform')
```

requirement for scikit-learn api:
- require data with no missing values
- in numpy array or pandas df
  pass numpy array with features in columns and obs in rows
- feature takes on continuous value
- target variable same number of obs as feature data

```
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party',axis=1)

# or in numpy array
y = df['party'].values
X = df.drop('party',axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
```

### Measuring model performance
- In classiffcation, accuracy is a commonly used metric
- Accuracy = Fraction of correct predictions
- Could compute accuracy on data used to fit classiffer
- NOT indicative of ability to generalize if predict on training data set
hence need to:  
    - Split data into training and test set
    - Fit/train the classifier on the training set
    - Make predictions on test set
    - Compare predictions with the known labels

### train test split
train and test sets are vital to ensure that your supervised learning model is able to generalize well to new data. 
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size=0.3,
    random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(\"Test set predictions:\\n {}\".format(y_pred))
Test set predictions:
[2 1 2 2 1 0 1 0 0 1 0 2 0 2 2 0 0 0 1 0 2 2 2 0 1 1 1 0 0
1 2 2 0 0 2 2 1 1 2 1 1 0 2 1]
knn.score(X_test, y_test)
0.9555555555555556
```

stratify so distribution same as training and testing data as they are in original dataset

### digits recognition dataset example
```
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits['DESCR'])

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```
```
# Import necessary modules
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
```

```
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train,y_train )

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

### additional
.reshape((-1,1))
output:(is a 1 dimensional columnar array)
.reshape((1,-1))
output:(is a 1 dimensional row array)

put into numpy array first

### Fitting a regression model
however most of the time, will use linear regression with regularization, not just like this
```
# Creating feature and target arrays
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values

# Predicting house value from a single feature
# since you are going to use only one feature to begin with, you need to do some reshaping using NumPy's .reshape()
X_rooms = X[:,5]
type(X_rooms), type(y)
(numpy.ndarray, numpy.ndarray)
# numpy array then reshape
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

# Plotting house value vs. number of rooms
plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show();

import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms),
    max(X_rooms)).reshape(-1, 1)
plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space),
    color='black', linewidth=3)
plt.show()
```

### Regression mechanics
y = ax + b
y = target
x = single feature
a, b = parameters of model

the loss function:
Ordinary least squares(OLS): Minimize sum of squares of
residuals
in sklearn, default is R-square: amt of var target variable predicted from feature variables

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)
```

```
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility,y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility,y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
```

```
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))
```

### Cross-validation motivation
Model performance is dependent on way the data is split
Not representative of the model’s ability to generalize
Solution: Cross-validation!
so avoid of problem of metric dependent on train test split

first fold will be hold out as test set, fit model on remaining folds as training set, then predict on the test set and compute metric of interest

More folds = More computationally expensive

### Cross-validation in scikit-learn
```
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
```

 can use %timeit to see how long cv takes

### Why regularize?
Linear regression minimizes a loss function
It chooses a coefficient for each feature variable
Large coefficients can lead to overfitting
Penalizing large coefficients: Regularization

### Ridge regression
large coefficient of either positive or negative is penalized by alpha 
Loss function = OLS loss function + α ∗ sum of (a[i])^2
Alpha: Parameter we need to choose
Picking alpha here is similar to picking k in k-NN
Hyperparameter tuning
Alpha controls model complexity
Alpha = 0: We get back OLS (Can lead to overfiting)
Very high alpha: Can lead to underfitting
This is L2 regularization.
```
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.3, random_state=42)
# make all variables in same scale when normalize = True
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
```


plotting r^2 for each alpha:
```
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
```

example:
```
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

```


### Lasso regression
Loss function = OLS loss function + α ∗ sum of ∣a[i]∣
This is also known as L1 regularization
```
from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.3, random_state=42)
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
```

benefits: 
Can be used to select important features of a dataset
Shrinks the coe cients of less important features to exactly 0
Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.

view feature importance:
```
from sklearn.linear_model import Lasso
names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()
```

### confusion matrix
accuracy is not always an informative metric. hence evaluating the performance of binary classifiers by computing a confusion matrix

### Metrics from the confusion matrix
Precision = positive predictive value = PPV = tp/ (tp+fp)
High precision: low false positive, not many real emails predicted as spam

Recall = sensitive = hit rate = true positive rate = tp / (tp+fn)
(slide got typo)
High recall: Predicted most spam emails correctly

recall and precision inverse relationship
- recall of 1 corresponds to a classifier with a low threshold in which all females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did not have diabetes.
- Precision and recall does not take true negatives into consideration.


F1score = (2*precision*recall)/(precision+recall)

Accuracy = (tp+tn)/(tp+tn+fp+fn)

### Class imbalance example: Emails
Spam classi cation
99% of emails are real; 1% of emails are spam
Could build a classi er that predicts ALL emails as real
99% accurate!
But horrible at actually classifying spam
Fails at its original purpose

### Confusion matrix in scikit-learn
```
# Import necessary modules
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,        test_size=0.4,random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
       precision    recall  f1-score   support

          0       0.77      0.85      0.81       206
          1       0.62      0.49      0.55       102

avg / total       0.72      0.73      0.72       308
```

The support gives the number of samples of the true response that lie in that class

### Logistic regression for binary classification
Logistic regression outputs probabilities
If the probability ‘p’ is greater than 0.5:
The data is labeled ‘1’
If the probability ‘p’ is less than 0.5:
The data is labeled ‘0’
produces Linear decision boundary
By default, logistic regression threshold = 0.5

```
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### ROC curve
Larger area under the ROC curve (known as AUC)= better model
area =1 means true positive =1, false positive =0

when threshold =0, model predicts '1' for all the data, meaning true positive and false positive rate =1,
when threshold =1, model predicts '0' for all the data, meaning true positive and false positive rate =0,

if vary threshold, get a set of points is called receiver operating characteristic curve = ROC curve


```
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();
# y-axis (True positive rate) is also known as recall
```
.predict_proba() method which returns the probability of a given sample being in a particular class. 
predict_proba returns array with two cols: each cols contain prob of each target value. second col is the one with index 1 meaning prob of being 1

### AUC in scikit-learn
auc is area under roc curve
```
from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)
```

### AUC using cross-validation
```
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv=5,
scoring='roc_auc')
print(cv_scores)
```

### AUC in scikit-learn and cross-val
```
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# scikit-learn method
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test,y_pred_prob)))

# cross val score method
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
```

****** notice in cross_val is use X and y. not X_train etc

### Choosing the correct hyperparameter
Try a bunch of different hyperparameter values
Fit all of them separately
See how well each performs
Choose the best performing one
It is essential to use cross-validation to avoid overfit to train set.

###  KNN with GridSearchCV in scikit-learn
```
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
knn_cv.best_params_
{'n_neighbors': 12}
knn_cv.best_score_
0.933216168717
```

###  logistic regression with GridSearchCV in scikit-learn
logistic regression also has a regularization parameter: C. C controls the inverse of the regularization strength, and this is what you will tune in this exercise. A large C can lead to an overfit model, while a small C can lead to an underfit model.
logistic regression also has a 'penalty' hyperparameter which specifies whether to use 'l1' or 'l2' regularization.

```
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

```

### Hyperparameter tuning with RandomizedSearchCV
GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use RandomizedSearchCV.

Note that RandomizedSearchCV will never outperform GridSearchCV. Instead, it is valuable because it saves on computation time.

### decision tree classifier with randomizedsearchcv
```
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
```

### Hold-out set reasoning
How well can the model perform on never before seen data?
Using ALL data for cross-validation is not ideal
1. Split data into training and hold-out set at the beginning using train_test_split
2. Perform grid search cross-validation on training set to tune model's hyperparameters
3. Choose best hyperparameters and evaluate on hold-out set which has not been used yet, to test how well model perform

### Elastic net regularization
the penalty term is a linear combination of the L1 and L2 penalties: a∗L1 + b∗L2
In scikit-learn, this term is represented by the 'l1_ratio' parameter: An 'l1_ratio' of 1 corresponds to an L1 penalty, and anything lower is a combination of L1 and L2.

```
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net,param_grid,cv=5)

# Fit it to the training data
gm_cv.fit(X_train,y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
```

### additional Assessing the Fit of Regression Models
A well-fitting regression model results in predicted values close to the observed data values. The mean model, which uses the mean for every predicted value, generally would be used if there were no informative predictor variables. The fit of a proposed regression model should therefore be better than the fit of the mean model.

Three statistics are used in Ordinary Least Squares (OLS) regression to evaluate model fit: R-squared, the overall F-test, and the Root Mean Square Error (RMSE). All three are based on two sums of squares: Sum of Squares Total (SST) and Sum of Squares Error (SSE). SST measures how far the data are from the mean, and SSE measures how far the data are from the model’s predicted values. Different combinations of these two values provide different information about how the regression model compares to the mean model.

#### R-squared and Adjusted R-squared

The difference between SST and SSE is the improvement in prediction from the regression model, compared to the mean model. Dividing that difference by SST gives R-squared. It is the proportional improvement in prediction from the regression model, compared to the mean model. It indicates the goodness of fit of the model.

R-squared has the useful property that its scale is intuitive: it ranges from zero to one, with zero indicating that the proposed model does not improve prediction over the mean model, and one indicating perfect prediction. Improvement in the regression model results in proportional increases in R-squared.

One pitfall of R-squared is that it can only increase as predictors are added to the regression model. This increase is artificial when predictors are not actually improving the model’s fit. To remedy this, a related statistic, Adjusted R-squared, incorporates the model’s degrees of freedom. Adjusted R-squared will decrease as predictors are added if the increase in model fit does not make up for the loss of degrees of freedom. Likewise, it will increase as predictors are added if the increase in model fit is worthwhile. Adjusted R-squared should always be used with models with more than one predictor variable. It is interpreted as the proportion of total variance that is explained by the model.

There are situations in which a high R-squared is not necessary or relevant. When the interest is in the relationship between variables, not in prediction, the R-square is less important. An example is a study on how religiosity affects health outcomes. A good result is a reliable relationship between religiosity and health. No one would expect that religion explains a high percentage of the variation in health, as health is affected by many other factors. Even if the model accounts for other variables known to affect health, such as income and age, an R-squared in the range of 0.10 to 0.15 is reasonable.

#### The F-test

The F-test evaluates the null hypothesis that all regression coefficients are equal to zero versus the alternative that at least one is not. An equivalent null hypothesis is that R-squared equals zero. A significant F-test indicates that the observed R-squared is reliable and is not a spurious result of oddities in the data set. Thus the F-test determines whether the proposed relationship between the response variable and the set of predictors is statistically reliable and can be useful when the research objective is either prediction or explanation.

### RMSE

The RMSE is the square root of the variance of the residuals. It indicates the absolute fit of the model to the data–how close the observed data points are to the model’s predicted values. Whereas R-squared is a relative measure of fit, RMSE is an absolute measure of fit. As the square root of a variance, RMSE can be interpreted as the standard deviation of the unexplained variance, and has the useful property of being in the same units as the response variable. Lower values of RMSE indicate better fit. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction.

The best measure of model fit depends on the researcher’s objectives, and more than one are often useful. The statistics discussed above are applicable to regression models that use OLS estimation. Many types of regression models, however, such as mixed models, generalized linear models, and event history models, use maximum likelihood estimation. These statistics are not available for such models.

### Dealing with categorical features in Python
scikit-learn: OneHotEncoder()
pandas: get_dummies()

also use boxplot to visualize categorical features
```
import pandas as pd
df = pd.read_csv('auto.csv')

# Create a boxplot of life expectancy per region
df.boxplot( 'life','Region', rot=60)

# Show the plot
plt.show()

# to avoid duplicate information. no need the extra feature
df_origin = pd.get_dummies(df)
df_origin = df_origin.drop('origin_Asia', axis=1)

# or just drop_first
# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df,drop_first=True)

```

### Dropping missing data
```
df = df.dropna()
df.shape
```

```
# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))
```

### Imputing missing data
Making an educated guess about the missing values
Example: Using the mean of the non-missing entries
```
from sklearn.preprocessing import Imputer
# axis=0 impute along columns
# axis=1 impute across rows
imp = Imputer(missing_values='NaN', strategy='mean', axis=0
imp.fit(X)
X = imp.transform(X)
```

imputers are known as transformers

#### Imputing within a pipeline
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
logreg = LogisticRegression()
steps = [('imputation', imp),
    ('logistic_regression', logreg)]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)
```

### why use NaN
We use NaN because it is an efficient and simplified way of internally representing missing data, and it lets us take advantage of pandas methods such as .dropna() and .fillna(), as well as scikit-learn's Imputation transformer Imputer()

### Imputing missing data in a ML Pipeline Example
an use the .fit() and .predict() methods on pipelines just as you did with your classifiers and regressors
```
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))

```

### Why scale your data?
- Many models use some form of distance to inform them
- Features on larger scales can unduly in uence the model
- Example: k-NN uses distance explicitly when making
predictions
- We want features to be on a similar scale
- Normalizing (or scaling and centering)

### Ways to normalize your data
- Standardization: Subtract the mean and divide by variance
- All features are centered around zero and have variance one
- Can also subtract the minimum and divide by the range 
- Minimum zero and maximum one
- Can also normalize so the data ranges from -1 to +1
- See scikit-learn docs for further details
- scaling can improve model
### Scaling in scikit-learn
```
from sklearn.preprocessing import scale
X_scaled = scale(X)
np.mean(X), np.std(X)
(8.13421922452, 16.7265339794)
np.mean(X_scaled), np.std(X_scaled)
(2.54662653149e-15, 1.0)
```

### Scaling in a pipeline
```
from sklearn.preprocessing import StandardScaler
steps = [('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)
0.956
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
knn_unscaled.score(X_test, y_test)
0.928
```
scaling improved model

### CV, scaling and GridsearchCV in a pipeline for Classification
```
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline,param_grid=parameters)

# Fit to the training set
cv.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

```

### Centering and scaling in a pipeline example
```
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train,y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test)))
```

### imputation,scaling,gridsearch in a pipeline for classification
```
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet',ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline,param_grid=parameters)

# Fit to the training set
gm_cv.fit(X_train,y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
```

## Introduction To Deep Learning In Python

### Neural networks 
- account for interactions really well which linear regression cannot
- is an powerful of deep learning model
- can be used for Text, Images, Videos, Audio, Source code
- have input layer, output layer, hidden layer
- hidden layer are not visible. cannot be observed directly in the world
- each dot in the hidden layer is called nodes
- nodes representations aggregation of information from input data 
- each node add to model's ability to capture interaction, more nodes more interation
- each data point is an obs
- The last layers capture the most complex interactions.


### Build and tune deep learning models using keras
```
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
predictors = np.loadtxt('predictors_data.csv', delimiter=',')
n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
```

### Forward propagation

```
import numpy as np
input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]),
    'node_1': np.array([-1, 1]),
    'output': np.array([2, -1])}

# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data*weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs*weights['output']).sum()

# Print output
print(output)

```

### activation function
- activation function allows model to capture non-linearity
- Applied to node inputs to produce node output
- applied to each node
- if no activation function can think of using identity function: returning the input

### ReLU (Rectified Linear Activation)
- an industrial standard activation function that leads to high performance networks
- relu(x)=0 if x<0, =x if x>=0. function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.
- applied tanh to convert input to output
```
import numpy as np
input_data = np.array([-1, 2])
weights = {'node_0': np.array([3, 3]),
    'node_1': np.array([1, 5]),
    'output': np.array([2, -1])}
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_input)
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_input)
hidden_layer_outputs = np.array([node_0_output, node_1_output])
output = (hidden_layer_output * weights['output']).sum()
print(output)
1.2382242525694254
```

```
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)
```

put everything into a function:
```
# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row*weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row*weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs*weights['output'])
    model_output = input_to_final_layer.sum()
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row,weights))

# Print results
print(results)
```

### multiple hidden layers
use forward propagation process but apply it iteratively more times
1. first fill in values for hidden layer one as a function of the inputs.
2. apply activation function to fill in the values in these nodes
3. use values from first hidden layer to fill in second hidden layer, etc
4. make prediction based on output of last hidden layer

### Representation learning
- deep learning sometimes also called representation learning
because subsequent layers build increasingly sophisticated
representations of raw data
- Deep networks internally build representations of patterns in
the data
- find increasingly complex patterns through successive hidden layers in the network
- Partially replace the need for feature engineering
- Modeler doesn't need to specify the interactions
- When you train the model, the neural network gets weights
that find the relevant patterns to make better predictions

### forward propagation for a neural network with 2 hidden layers
```
def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data*weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs*weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs*weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = relu((hidden_1_outputs*weights['output']).sum())
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)
```

### to model weights for optimization
- use loss function to Aggregates errors in predictions from many data points into single number: Measure of model's predictive performance
- Lower loss function value means a better model
- Goal: Find the weights that give the lowest value for the loss
function
- use Gradient descent: use slope (derivative) to find the 
minimum value

### illustration how how weight changes affect accuracy
```
# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data,weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)
```

```
from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row,weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row,weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(model_output_0,target_actuals)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(model_output_1,target_actuals)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)
```

### Gradient descent
=slope
- If the slope is positive:
    - Going opposite the slope means moving to lower numbers
    - Subtract the slope from the current value
    - Too big a step might lead us astray
- Solution: learning rate
    - Update each weight by subtracting learning rate * slope

#### To calculate the slope for a weight
need to multiply:
- Slope of the loss function w.r.t value at the node we feed
into
- The value of the node that feeds into our weight
- Slope of the activation function w.r.t value we feed into

Code to calculate slopes and update weights:
```
import numpy as np
weights = np.array([1, 2])
input_data = np.array([3, 4])
target = 6
learning_rate = 0.01
preds = (weights * input_data).sum()
error = preds - target
print(error)
5
gradient = 2 * input_data * error
gradient
array([30, 40])
weights_updated = weights - learning_rate * gradient
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print(error_updated)
2.5
```

```
# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights-learning_rate*slope

# Get updated predictions: preds_updated
preds_updated = (input_data*weights_updated).sum()

# Calculate updated error: error_updated
error_updated = preds_updated-target

# Print the original error
print(error)

# Print the updated error
print(error_updated)
```

To keep your code clean, there is a pre-loaded get_slope() function that takes input_data, target, and weights as arguments. There is also a get_mse() function that takes the same arguments. The input_data, target, and weights have been pre-loaded.
```
n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01* slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
```

### Backpropagation 
- takes error from output layer and propagates it backward through hidden layers towards input layer
- Allows gradient descent to update all weights in neural
network (by gettng gradients for all weights)
- Comes from chain rule of calculus
- Trying to estimate the slope of the loss function w.r.t each
weight
- Do forward propagation to calculate predictions and errors before doing backpropagation
- Go back one layer at a time
- Gradients for weight is product of:
    1. Node value feeding into that weight
    2. Slope of loss function w.r.t node it feeds into
    3. Slope of activation function at the node it feeds into
- Need to also keep track of the slopes of the loss function w.r.t node values
- Slope of node values are the sum of the slopes for all weights
that come out of them
- Each time you generate predictions using forward propagation, you update the weights using backward propagation.

### steps for backpropagation in neural network
1. Start at some random set of weights
2. Use forward propagation to make a prediction
3. Use backward propagation to calculate the slope of the loss
function w.r.t each weight
4. Multiply that slope by the learning rate, and subtract from
the current weights
5. Keep going with that cycle until we get to a flat part

### Stochastic gradient descent
- It is common to calculate slopes on only a subset of the data
(a batch)
- Use a different batch of data to calculate the next update
- Start over from the beginning once all data is used
- Each time through the training data is called an epoch
- When slopes are calculated on one batch at a time its called
stochastic gradient descent

### keras workflow: model building
1. . Specify Architecture
    - how many layers 
    - how many nodes
    - what activation function to use in each layer
2. Compile
    - loss function
3. Fit  
    - cycle of backpropagation and optimization of model weights with data
4. Predict


### Sequential models
sequential models require that each layer has weights or connections only to the one layer coming directly after it in the network diagram

there are more complex models. sequential easiest

```
import numpy as np
# called Dense because all the nodes in the previous layer connect to all of the nodes in the current layer
# possible to have non-Dense models
from keras.layers import Dense
from keras.models import Sequential

# read data
predictors = np.loadtxt('predictors_data.csv', delimiter=',')
# store number of cols important for building keras model
# because = number of nodes in input layer
n_cols = predictors.shape[1]

model = Sequential()

# first layer, need specify input shape
# (n_cols,) means will have n_cols columns and can be any number of rows/data points
# but this is also the hidden layer
# 100 is the number of nodes 
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))

model.add(Dense(100, activation='relu'))

# output layer one node
model.add(Dense(1))

# total this model got 2 hidden layer
```

```
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32,activation='relu'))

# Add the output layer
model.add(Dense(1))
```

### Compile model
1. Specify the optimizer: which controls the learning rate
    - good learning rate can mean finding model faster and may also affect how good a set of weights
    - Many options and mathematically complex
    - "Adam" is usually a good choice as a optimizer.
2. Specify Loss function
    - 'mean_squared_error' common for regression, 'categorical_crossentropy' common for classification
    - keras for classification will have a new default metric cover later

add compile after building model:
```
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

### Fitting a model
- Applying backpropagation and gradient descent with your
data to update the weights
- Scaling data before fitting can ease optimization
    - one common approach is subtract each feature by its mean and divide by sd.

```
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(predictors, target)
```

### Neural network - Classification
- 'categorical_crossentropy' loss function
- Similar to log loss: Lower is be er
- Add metrics = ['accuracy'] to compile step to see the accuracy (what fraction of predictions were correct) at the end of each epoch.
- Output layer has separate node for each possible outcome,
and uses 'softmax' activation

```
from keras.utils.np_utils import to_categorical
data = pd.read_csv('basketball_shot_log.csv')
# drop target column and store as numpy matrix
predictors = data.drop(['shot_result'], axis=1).as_matrix()
# one hot encoding
target = to_categorical(data.shot_result)
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
# for classification
# 2 nodes, means 2 possible outcome
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(predictors, target)
```

### Saving, reloading and using your Model
models are saved in format hdf5, which h5 is the common extension.

```
from keras.models import load_model
model.save('model_file.h5')
my_model = load_model('my_model.h5')
predictions = my_model.predict(data_to_predict_with)
# only want prob that the class is 1 so get second column with numpy indexing
probability_true = predictions[:,1]
# see model architecture
my.model.summary()
```

### Why optimization is hard
- Simultaneously optimizing 1000s of parameters with complex
relationships
- Updates may not improve model meaningfully even with good optimizer like 'adams'
- Updates too small (if learning rate is low) or too large (if
learning rate is high)
- sometimes best to use the simplest optimizer: sgd

### Stochastic gradient descent (Sgd)
```
def get_new_model(input_shape = input_shape):
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
return(model)

lr_to_test = [.000001, 0.01, 1]

# loop over learning rates
for lr in lr_to_test:
model = get_new_model()
my_optimizer = SGD(lr=lr)
model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
model.fit(predictors, target)
```

### Dying Neuron Problem 
neuron takes a value less than 0 for all rows of data.
example can be due to negative value is node for relu activation where slope=0. weight dont get updated.

can use activation function whose slope is never exactly zero.

### Vanishing Gradient Problem
HOWEVER, an s shaped function tanh, values near the middle of the s, have very small slopes close to 0. repeated multiplication for deep network will make backprop updates become close to 0 too.
This instead creates Vanishing Gradient Problem.

Changing the activation function may be the solution that isnt even close to flat anyway. there are variations of relu although relu still more popular unless model isnt training better.

### Validation in deep learning
use validation data to test model performance
- Commonly use validation split rather than cross-validation
- Deep learning widely used on large datasets. computationally expensive to run k-fold cv
- Single validation score is based on large amount of data, and
is reliable
- Repeated training from cross-validation would take long time

```
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target, validation_split=0.3)
```

### Early Stopping
number of epochs model can go without improving before stop training. optimizations stops when its not improving

patience = 2 or 3 is a good enough number of epoch
```
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)

# argument nb_epoch renamed to epochs
model.fit(predictors, target, validation_split=0.3, epochs=20,
callbacks = [early_stopping_monitor])
```

got more advance callbacks. basic earlystopping good enough for now. default keras 10 epochs. since got early stopping, can increase number of epochs.

note callbacks takes list.

### plot results of two models in neural network

```
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100,activation='relu'))

# Add the output layer
model_2.add(Dense(2,activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
# r is red, b is blue
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
```

### model capacity
is a model's ability to capture predictive patterns in the data

increase model capacity (training error decreases but test sample error may increase)
- increase number of nodes/neurons in hidden layer
- add layers

#### Workflow for optimizing model capacity
- Start with a small network
- Gradually increase capacity
- Keep increasing capacity until validation score is no longer
improving

### digit recognition model
output layer 10 notes cause 10 digits
```
# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50,activation='relu',input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50,activation='relu'))

# Add the output layer
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(X,y,validation_split=0.3)
```

### get more datasets
wikipedia: list of datasets for machine learning research

### continue learning tips
- Start with standard prediction problems on tables of numbers
- Images (with convolutional neural networks) are common next steps
- keras.io for excellent documentation
- Graphical processing unit (GPU) provides dramatic speedups in model training times
- Need a CUDA compatible GPU
- For training on using GPUs in the cloud look here: http://bit.ly/2mYQXQb

## Unsupervised learning in python

dimensions = features

samples in this course is in numpy array

### k-means clustering
- Finds clusters of samples
- Number of clusters must be specified
- Implemented in sklearn ("scikit-learn")

Cluster labels for new samples
- New samples can be assigned to existing clusters
- k-means remembers the mean of each cluster (the
"centroids")
- Finds the nearest centroid to each new sample

```
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
KMeans(algorithm='auto', ...)
labels = model.predict(samples)
print(labels)
[0 0 1 1 0 1 2 1 0 1 ...]
# cluster labels for new samples
new_labels = model.predict(new_samples)
print(new_labels)
```

```
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)
```

### inspecting clustering
```
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,alpha=0.5,c=labels)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()
```

### Crosstab of labels and species
```
df = pd.DataFrame({'labels': labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
```

```
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)
```

### Inertia measures clustering quality
Measures how spread out the clusters are (lower is better)
Distance from each sample to centroid of its cluster
After fit() , available as a attribute inertia_
k-means attempts to minimize the inertia when choosing
clusters
However, more clusters means lower inertia. A good clustering has tight clusters (so low inertia) but not too many clusters!
Choose an "elbow" in the inertia plot: where inertia begins to decrease more slowly
```
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)       
```

### Example of KMeans clustering searching best number of clusters hyperparameter
```
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model=KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```

### StandardScaler
- In kmeans: feature variance = feature influence
- StandardScaler transforms each feature to have mean 0 and
variance 1
- Features are said to be "standardized"
- scaling can improve prediction 

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)
```

### StandardScaler and KMeans
```
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
labels = pipeline.predict(samples)
```


```
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels,'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['species'])

# Display ct
print(ct)
```

### sklearn preprocessing steps
- StandardScaler is a "preprocessing" step
- MaxAbsScaler and Normalizer are other examples

### Normalizer
While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, Normalizer() rescales each sample - here, each company's stock price - independently of the other.

### Normalizer and KMeans

eg. Which stocks move together?
```
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))
```
### Visualization with hierarchical clustering and t-SNE
"t-SNE" : Creates a 2D map of a dataset. maps the data samples into 2d space so that the proximity of the samples to one another can be visualized.
"Hierarchical clustering": merges the data samples into ever-coarser clusters, yielding a tree visualization of the resulting cluster hierarchy. 

### Dendrogram
tree-like diagram visualization of hierarchical clustering

Dendrograms show cluster distances
Height on dendrogram =
distance between merging
clusters

### "agglomerative" hierarchical clustering
Every country begins in a separate cluster
At each step, the two closest clusters are merged.
Continue until all countries in a single cluster

Height on dendrogram specifies max. distance between merging clusters
Don't merge clusters further apart than this (e.g. 15)

Distance between clusters:
- Defined by a "linkage method"
- In "complete" linkage: distance between clusters is max. distance between their samples
- Specified via method parameter, e.g. linkage(samples,method="complete")
- Different linkage method, different hierarchical clustering
- In complete linkage, the distance between clusters is the distance between the furthest points of the clusters. In single linkage, the distance between clusters is the distance between the closest points of the clusters.

linkage function that performs hierarchical clustering

linkage method defines how the distance between clusters is measured. In complete linkage, the distance between clusters is the distance between the furthest points of the clusters. In single linkage, the distance between clusters is the distance between the closest points of the clusters.

```
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: 
# or method='single
mergings = linkage(samples,method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
```

With 5 data samples, there would be 4 merge operations, and with 6 data samples, there would be 5 merges, and so on.

### normalize with hierarchical clustering
normalize is standalone cannot use in pipeline because cause cannot .fit, .fit_transform, .transform.

Normalizer is a transformer hence can use in pipeline. 

both normalize and normalizer transform the same way


```
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements,method='complete')

# Plot the dendrogram
dendrogram(mergings,labels=companies,leaf_rotation=90,leaf_font_size=6)
plt.show()
```

### Extracting cluster labels using fcluster
in intermediate stages
Use the fcluster() function
Returns a NumPy array of cluster labels

```
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method='complete')
from scipy.cluster.hierarchy import fcluster
# 15 is height
labels = fcluster(mergings, 15, criterion='distance')
print(labels)
```

### t-SNE 
- t-SNE = "t-distributed stochastic neighbor embedding"
- Maps samples to 2D space (or 3D)
- Map approximately preserves nearness of samples
- Great for inspecting datasets
- is a powerful tool for visualizing high dimensional data.
- t-SNE features different when run everytime but clusters position relative to one another are same

in sklearn:
only have fit_transform. no fit and transform separate. so cannot just transform for new samples. must restart.

learning rate try between 50 to 200.
```
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()
```

```
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

# Annotate the points
# Label each point with its company name
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
```

### Dimension reduction
Dimension reduction summarizes a dataset using its common occuring patterns. I
- finds patterns in data and re-express these patterns in a compressed form leading to more efficient storage and computation
- Remove less-informative "noise" features which cause problems for prediction tasks, e.g.
classi,cation, regression
- example of dimension reduction technique: PCA

### Principal Component Analysis
PCA = "Principal Component Analysis"
Fundamental dimension reduction technique
PCA is often used before supervised learning to improve model performance and generalization. It can also be useful for unsupervised learning 

1. First step "decorrelation": dosent change dimension of data. PCA aligns data with axes
Rotates data samples to be aligned with axes
Shifts data samples so they have mean 0
No information is lost. Due to rotation, 'de-correlates' the data. Resulting PCA features are not linearly correlated.
("decorrelation")
2. Second step reduces dimension 


PCA is a scikit-learn component 
fit() learns the transformation from given data
transform() applies the learned transformation
transform() can also be applied to new data

"Principal components" = directions of variance
PCA aligns principal components with the axes

principal components: pca.components_ 
- Each row defines displacement from mean principal 
- components are the directions along which the the data varies.

PCA features are in decreasing order of variance
Assumes the low variance features are "noise" and high variance features are informative. PCA discards low variance PCA features

PCA can dont need specify number of components but NMF (another technique) needs.
```
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)
transformed = model.transform(samples)
# numpy array with one row for each principle component
print(model.components_)

```

Rows of transformed correspond to samples
Columns of transformed are the "PCA features"

#### get transformed features
```
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)
```

#### scatterplot of the 2 pca features
E.g. PCA(n_components=2) means keeps the first 2 PCA features. Intrinsic dimension is a good choice
```
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(samples)
print(transformed.shape)

# scatterplot of the 2 pca features
import matplotlib.pyplot as plt
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()
```

#### longitude and latitude
Intrinsic dimension of a flight path
- 2 features: longitude and latitude at points along a
flight path
- Dataset appears to be 2-dimensional but can approximate using one feature: displacement along flight path
- Is intrinsically 1-dimensional

#### plotting variance of pca features using bar chart to decide intrinsic dimension:
```
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)
# enumerating pca features
features = range(pca.n_components_)
# Plotting the variances of PCA features
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
```

#### plotting first PCA component:
```
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
```

#### Intrinsic dimension
 = number of features needed to
approximate the dataset/ how much data can be compressed.
can be represented by PCA

PCA identifies intrinsic dimension when samples have any number of features
Intrinsic dimension = number of PCA features with significant variance

#### standardscaler and pca and decide intrinsic dimension
```
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features,pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
```

Since PCA features 0 and 1 have significant variance, the intrinsic dimension of this dataset appears to be 2.


###  Pearson correlation
- Measures linear correlation of features
- Value between -1 and 1
- Value of 0 means no linear correlation

decorrelating features with pca:
```
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width,length)

# Display the correlation
print(correlation)
```

after PCA no correlation:
```
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
```


### Word frequency arrays
Rows represent documents, columns represent words
Entries measure presence of each word in each document measure using "tf-idf"

### Sparse arrays and csr_matrix
"Sparse": most entries are zero
Can use scipy.sparse.csr_matrix instead of NumPy array
csr_matrix remembers only the non-zero entries 

### tf-idf word-frequency array
Measure presence of words in each document using "tf-idf"
"tf" = frequency of word in document
"idf" = is a weighting scheme that reduces influence of frequent words like 'the'

```
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)
```

### TruncatedSVD and csr_matrix
scikit-learn PCA doesn't support csr_matrix
Use scikit-learn TruncatedSVD instead
Performs same transformation
```
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents) # documents is csr_matrix
TruncatedSVD(algorithm='randomized', ... )
transformed = model.transform(documents)
```

### combine: pipeline, truncatedsvd, kmeans
```
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))

```

### Non-negative matrix factorization
NMF = "non-negative matrix factorization"
dimension reduction technique called "Non-negative matrix factorization" ("NMF") that expresses samples as combinations of interpretable parts
- Dimension reduction technique
- NMF models are interpretable (unlike PCA)
- Easy to interpret means easy to explain!
- However, all sample features must be non-negative (>= 0)!!!!!!
- Using scikit-learn NMF
- Follows fit() / transform() pa ern
- Must specify number of components unlike PCA e.g.
NMF(n_components=2)
- Works with NumPy arrays and with csr_matrix
- Sample can be reconstructed:
Multiply components by feature values, and add up
Can also be expressed as a product of matrices
This is the "Matrix Factorization" in "NMF"

2 examples of interpretable parts:
1. For documents:
NMF components represent topics
NMF features combine topics into documents
NMF expresses documents as combinations of topics (or "themes"). choosing words with highest values in each components will fit the "theme".
2. For images:
NMF components are parts of images
NMF expresses images as combinations of patterns encoded in arrays


```
# samples is the word-frequency array

from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)

# NMF feature values are non-negative
# Can be used to reconstruct the samples combine feature values with components
nmf_features = model.transform(samples)
print(nmf_features)

# NMF has components just like PCA has principal components
# Dimension of components = dimension of samples
# Entries are non-negative
print(model.components_)
```

```
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features,index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway',:])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington',:])

# Notice that for both actors, the NMF feature 3 has by far the highest value. This means that both articles are reconstructed using mainly the 3rd NMF component. NMF components represent topics (for instance, acting!).
```

#### visualizing images
Encode as 2D array can apply NMF!
Each row corresponds to an image
Each column corresponds to a pixel

```
#Visualizing samples
print(sample)
bitmap = sample.reshape((2, 3))
print(bitmap)
from matplotlib import pyplot as plt
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.show()
```

#### investigate topics of documents
```
# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_,columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3,:]

# Print result of nlargest
# gives the five words with the highest values for that component.
print(component.nlargest())
```

#### led digits dataset
use NMF to decompose grayscale images into their commonly occurring patterns

Firstly, explore the image dataset and see how it is encoded as an array. You are given 100 images as a 2D array samples, where each row represents a single 13x8 image. 
```
# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape((13,8))

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
```
 
##### displays the image encoded by any 1D array:
This time, you are also provided with a function show_as_image()
```
def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

```

```
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
# a led digit has 7 cells
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)
```

#### building recommender systems using NMF

given articles is a word frequency array

##### Cosine similarity
used to evaluate similarity between articles
Uses the angle between the lines (in pca)
Higher values means more similar
Maximum value is 1, when angle is 0 degrees
```
from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)

# Calculating the cosine similarities without using dataframe
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# if has index 23
current_article = norm_features[23,:]
similarities = norm_features.dot(current_article)
print(similarities)

# Calculating the cosine similarities using dataframe
from sklearn.preprocessing import normalize
import pandas as pd
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_article = df.loc['Dog bites man']
similarities = df.dot(current_article)
print(similarities.nlargest())
```

The NMF features you obtained earlier are available as nmf_features, while titles is a list of the article titles.
```
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo',:]

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
```

#### Recommender musical artists

sparse array artists whose rows correspond to artists and whose columns correspond to users. The entries give the number of times each artist was listened to by each user.

resulting normalized NMF features for recommendation:
```
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
# MaxAbsScaler, transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to.
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,nmf,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen',:]

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
```

## Machine Learning with Tree-Based Models in Python

### Classification and Regression Trees (CART) 
are a set of supervised learning models used for problems involving classification and regression.

### Classification-tree (Decision Tree Classifier)
- Sequence of if-else questions about individual features.
- Objective: infer class labels.
- Able to capture non-linear relationships between features and labels.
- Don't require feature scaling (ex:Standardization, ..)
- Decision region: region in the feature space where all instances are assigned to one class
label.
- Decision Boundary: surface separating different decision regions.

Decision-Tree Classifier
data structure consisting of a hierarchy of nodes.
Node: question or prediction.

Three kinds of nodes:
Root: no parent node, question giving rise to two children nodes.
Internal node: one parent node, question giving rise to two children nodes.
Leaf: one parent node, no children nodes --> prediction.

Criteria to measure the impurity of a node I(node):
gini index,
entropy. 
Most of the time, the gini index and entropy lead to the same results. The gini index is slightly faster to compute and is the default criterion used 

Nodes are grown recursively.
At each node, split the data based on:
feature and split-point to maximize IG(node). (IG = information gain)
If IG(node)= 0, declare the node a leaf.

interesting dataset: Breast Cancer Wisconsin (Diagnostic)
Predict whether the cancer is benign or malignant
https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

```
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2,
stratify=y,
random_state=1)
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
# Fit dt to the training set
dt.fit(X_train,y_train)
# Predict test set labels
y_pred = dt.predict(X_test)
# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
```

```
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2,
stratify=y,
random_state=1)
# Instantiate dt, set 'criterion' to 'gini'
dt = DecisionTreeClassifier(criterion='gini', random_state=1)

# Fit dt to the training set
dt.fit(X_train,y_train)
# Predict test-set labels
y_pred= dt.predict(X_test)
# Evaluate test-set accuracy
accuracy_score(y_test, y_pred)
```

example:
```
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)
```

### Logistic regression vs classification tree
A classification tree divides the feature space into rectangular regions. In contrast, a linear model such as logistic regression produces only a single linear decision boundary dividing the feature space into two decision regions.

### Decision-Tree for Regression
impurity of a node = MSE of the targets in that node.
The regression trees tries to find the splits that produce leafs where in each leaf the target values are on average, the closest possible to the mean_value of the labels in that particular leaf

```
# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2,
random_state=3)
# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,
min_samples_leaf=0.1,
random_state=3)
# Fit 'dt' to the training-set
dt.fit(X_train, y_train)
# Predict test-set labels
y_pred = dt.predict(X_test)
# Compute test-set MSE
mse_dt = MSE(y_test, y_pred)
# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)
# Print rmse_dt
print(rmse_dt)
```

```
# Fit 'dt' to the training-set
dt.fit(X_train, y_train)

# Predict test-set labels
y_pred = dt.predict(X_test)

# Compute test-set MSE
mse_dt = MSE(y_test, y_pred)

# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print(rmse_dt)

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

# instantiated a linear regression model lr and trained it on the same dataset as dt.
# Linear regression vs regression tree
# Predict test set labels 
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_pred_lr, y_test)

# Compute rmse_lr
rmse_lr = mse_lr**(1/2)

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
```

#### Advantages of CARTs
- Simple to understand.
- Simple to interpret.
- Easy to use.
- Flexibility: ability to describe non-linear dependencies.
- Preprocessing: no need to standardize or normalize features

#### Limitations of CARTs
- Classiffication: can only produce orthogonal decision boundaries.
- Sensitive to small variations in the training set.
- High variance: unconstrained CARTs may overfit the training set.
- Solution: ensemble learning.

### Goals of Supervised Learning
Find a model that best approximates f_hat : ≈ f
End goal: f_hat should acheive a low predictive error on unseen datasets.

#### Difficulties in Approximating f
Overfitting: f_hat(x) fits the training set noise.
Underfitting: f_hat is not flexible enough to approximate f .

#### Generalization Error 
= bias + variance + irreducible error

Bias: error term that tells you, on average, how much f_hat ≠ f.
Variance: tells you how much is f_hat inconsistent over different training sets.
Irreducible error is instead regardless of model complexity

Estimating the Generalization Error
Cannot be done directly because:
- f is unknown,
- usually you only have one dataset,
- noise is unpredictable.

Solution: cross-validation
split the data to training and test sets,
fit f_hat to the training set,
evaluate the error of f_hat on the unseen test set.
generalization error of f_hat ≈ test set error of f_hat

can use K-fold CV or Hold-out CV on training set

### K-fold CV
keep the test set untouched until you are confident about your model's performance. 

for K=10: 
- First, the training set (T) is split randomly into 10 partitions or folds, 
- The error of fhat is evaluated 10 times on the 10 folds, 
- Each time, one fold is picked for evaluation after training fhat on the other 9 folds. 
- At the end, you'll obtain a list of 10 errors
- CV-error is computed as the mean of the 10 obtained errors.

If f_hat suffers from high variance: CV error of f_hat > training set error of f_hat,
f_hat is said to overfit the training set. To remedy decrease model complexity or gather more data

if f_hat suffers from high bias: CV error of f_hat ≈ training set error of f_hat but >> desired error.
is said to underfit the training set. To remedy: increase model complexity,
gather more relevant features

```
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
# Set seed for reproducibility
SEED = 123
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
test_size=0.3,
random_state=SEED)
# Instantiate decision tree regressor and assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4,
min_samples_leaf=0.14,
random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train, cv= 10,
scoring='neg_mean_squared_error',
n_jobs = -1)
# Fit 'dt' to the training set
dt.fit(X_train, y_train)
# Predict the labels of training set
y_predict_train = dt.predict(X_train)
# Predict the labels of test set
y_predict_test = dt.predict(X_test)

# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
CV MSE: 20.51
# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))
Train MSE: 15.30
# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))


# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
CV MSE: 20.51
# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))
Train MSE: 15.30
# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))
Test MSE: 20.92
```
Given that the training set error is smaller than the CV-error, we can deduce that dt overfits the training set and that it suffers from high variance. Notice how the CV and test set errors are roughly equal.

another example:
```
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)


# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt,X_train, y_train, cv=10, 
                       scoring='neg_mean_squared_error',
                       n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

#  Notice how the training error is roughly equal to the 10-folds CV error you obtained earlier
```


#### Model Complexity
sets the flexibility of f_hat
Example: Maximum tree depth, Minimum samples per leaf

best complexity is lowest generalization error. (bias_variance tradeoff)

As the complexity of f_hat increases, the bias term decreases while the variance term increases.


### boosting
Boosting refers to an ensemble method in which several models are trained sequentially with each model learning from the errors of its predecessors. In this chapter, you'll be introduced to the two boosting methods of AdaBoost and Gradient Boosting.

### Ensemble Learning
- Train different models on the same dataset.
- Let each model make its predictions.
- Meta-model: aggregates predictions of individual models.
- Final prediction: more robust and less prone to errors.
- Best results: models are skillful in different ways.
- First, the training set is fed to different classifiers. Each classifier learns its parameters and makes predictions. Then these predictions are fed to a meta model which aggregates them and outputs a final prediction.

#### Voting Classifier
Voting Classifier in sklearn (Breast-Cancer dataset)

```
# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size= 0.3,
random_state= SEED)
# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
('K Nearest Neighbours', knn),
('Classification Tree', dt)]

# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
#fit clf to the training set
clf.fit(X_train, y_train)
# Predict the labels of the test set
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) 
# Evaluate clf's accuracy on the test set
print('{:s} : {:.3f}'.format(clf_name, accuracy))

# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)
# Fit 'vc' to the traing set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))

```

#### Bagging / Bootstrap aggregation 
- is an ensemble method involving training the same algorithm many times using different subsets sampled from the training data via sample with replacement
- Reduces variance of individual models in the ensemble


BaggingClassifier: 
- Aggregates predictions by majority voting.
BaggingRegressor: 
- Aggregates predictions through averaging.

```
# Import models and utility functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
stratify=y,
random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, random_state=1,n_jobs=-1)
# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)
# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))
```

#### Voting Classiffier vs Bagging:
same training set vs differetn subsets of the training set
different algorithms vs one algorithm

#### Out Of Bag Evaluation
Out Of Bag (OOB) instances
- On average, for each model, 63% of the training instances are sampled.
- The remaining 37% constitute the OOB instances.
- Since OOB instances are not seen by a model during training, these can be used to estimate the performance of the ensemble without the need for cross-validation. This technique is known as OOB-evaluation.


```
# Import models and split utility function
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,
stratify= y,
random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4,
min_samples_leaf=0.16,
random_state=SEED)
# Instantiate a BaggingClassifier 'bc'; set oob_score= True
bc = BaggingClassifier(base_estimator=dt, n_estimators=300,
oob_score=True, n_jobs=-1)
# Fit 'bc' to the traing set
bc.fit(X_train, y_train)
# Predict the test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
# Extract the OOB accuracy from 'bc'
oob_accuracy = bc.oob_score_
# Print test set accuracy
print('Test set accuracy: {:.3f}'.format(test_accuracy))
# Print OOB accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy))

```
Note that in scikit-learn, the OOB-score corresponds to the accuracy for classifiers and the r-squared score for regressors. 

These results highlight how OOB-evaluation can be an efficient technique to obtain a performance estimate of a bagged-ensemble on unseen data without performing cross-validation.

#### Random Forests
is an ensemble method.
further ensemble diversity through randomization at the level of each split in the trees forming the ensemble.

RandomForestClassifier:
Aggregates predictions by majority voting
RandomForestRegressor:
Aggregates predictions through averaging

in general, RF achieves lower variance than individual trees.

Random Forests Hyperparameters:
CART hyperparameters
number of estimators
bootstrap

##### Feature Importance
Tree-based methods: enable measuring the importance of each feature in prediction.
In sklearn :
- how much the tree nodes use a particular feature to reduce impurity, expressed as a percentage indicating the weight of that feature in training and prediction
- accessed using the attribute feature_importance_

```
import pandas as pd
import matplotlib.pyplot as plt
# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()
```

#### Bagging vs RandomForests:
Bagging:
- Base estimator: Decision Tree, Logistic Regression, Neural Net, etc
- Each estimator is trained on a distinct bootstrap sample of the training set
- Estimators use all features for training and prediction

Random Forests:
- Base estimator: Decision Tree
- Each estimator is trained on a different bootstrap sample having the same size as the training set
- RF introduces further randomization in the training of individual trees
- d features are sampled at each node without replacement
( d < total number of features )
d refers to square-root of the number of features in scikit-learn.

```
# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3,
random_state=SEED)

# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,
min_samples_leaf=0.12,
random_state=SEED)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

```

#### Boosting
- Ensemble method combining several weak learners to form a strong learner.
- Weak learner: Model doing slightly better than random guessing.
- Example of weak learner: Decision stump (CART whose maximum depth is 1).
- Train an ensemble of predictors sequentially.
- Each predictor pays more attention to the instances wrongly predicted by its predecessor by constantly changing the weights of training instances.
- Most popular boosting methods: AdaBoost, Gradient Boosting.

##### Adaboost
- Stands for Adaptive Boosting.
- Each predictor pays more attention to the instances wrongly predicted by its predecessor.
- Achieved by changing the weights of training instances.
- individual predictors need not to be CARTs. However CARTs are used most of the time in boosting because of their high variances
- Each predictor is assigned a coefficient α.
- α depends on the predictor's training error
- Learning rate: 0 < η ≤ 1
it is used to shrink the coefficient alpha of a trained predictor. It's important to note that there's a tradeoff between the learning rate and the number of estimators. A smaller value of learning rate should be compensated by a greater number of estimators.

AdaBoostClassifier: Weighted majority voting.
AdaBoostRegressor: Weighted average.

```
# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
stratify=y,
random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
# Instantiate an AdaBoost classifier 'adab_clf'
# 100 decision stumps
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)
# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
```

##### Gradient Boosted Trees
- Sequential correction of predecessor's errors.
- Does not tweak the weights of training instances.
- Fit each predictor is trained using its predecessor's residual errors as labels.
- Gradient Boosted Trees: a CART is used as a base learner.
- In sklearn: GradientBoostingRegressor,
GradientBoostingClassifier
- Cons:
GB involves an exhaustive search procedure.
Each CART is trained to find the best split points and features.
May lead to CARTs using the same split points and maybe the same features. to mitagte the problem: Stochastic gradient boosting

```
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
test_size=0.3,
random_state=SEED)

# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)
# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)
# Predict the test set labels
y_pred = gbt.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test))
```

##### Shrinkage
An important parameter used in training gradient boosted trees is shrinkage. shrinkage refers to the fact that the prediction of each tree in the ensemble is shrinked after it is multiplied by a learning rate which is a number between 0 and 1. Similarly to AdaBoost, there's a trade-off between learning rate and the number of estimators. Decreasing the learning rate needs to be compensated by increasing the number of estimators in order for the ensemble to reach a certain performance.

#### Stochastic Gradient Boosting (SGB)
- Each tree is trained on a random subset of rows of the training data.
- The sampled instances (40%-80% of the training set) are sampled without replacement.
- Features are sampled (without replacement) when choosing split points.
- Result: further ensemble diversity.
- Effect: adding further variance to the ensemble of trees.
- Once a tree is trained, predictions are made and the residual errors can be computed. These residual errors are multiplied by the learning rate and are fed to the next tree in the ensemble. This procedure is repeated sequentially until all the trees in the ensemble are trained. 

```
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
test_size=0.3,
random_state=SEED)

# Instantiate a stochastic GradientBoostingRegressor 'sgbt'
# max_features is the number of features to consider when looking for the best split
sgbt = GradientBoostingRegressor(max_depth=1,
subsample=0.8,
max_features=0.2,
n_estimators=300,
random_state=SEED)
# Fit 'sgbt' to the training set
sgbt.fit(X_train, y_train)
# Predict the test set labels
y_pred = sgbt.predict(X_test)

# Evaluate test set RMSE 'rmse_test'
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print 'rmse_test'
print('Test set RMSE: {:.2f}'.format(rmse_test))
```

### Hyperparameters
parameters: learned from data
hyperparameters: not learned from data, set prior to training
optimal hyperparameters are those of the model achieving the best CV score

Approaches to hyperparameter tuning:
- Grid Search
- Random Search
- Bayesian Optimization
- Genetic Algorithms
etc

summary:
.best_score_ : best CV score
.best_estimator_ : extract best model
.best_params_ : extract best hyperparameters
.predict_prob(X_test)[:,1] : probability for positive class
.predict(X_test) : y_pred
.score : averate score


accuracy:
```
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Set seed to 1 for reproducibility
SEED = 1
# Instantiate a DecisionTreeClassifier 'dt'
dt = DecisionTreeClassifier(random_state=SEED)

# Print out 'dt's hyperparameters
print(dt.get_params())

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Define the grid of hyperparameters 'params_dt'
params_dt = {
'max_depth': [3, 4,5, 6],
'min_samples_leaf': [0.04, 0.06, 0.08],
'max_features': [0.2, 0.4,0.6, 0.8]
}
# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt,
param_grid=params_dt,
scoring='accuracy',
cv=10,
n_jobs=-1)
# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)

# Extract best hyperparameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyerparameters:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format(best_CV_score))

# Extract best model from 'grid_dt'
# this model is fitted on whole training set because refit parameter of GridSearchCV is set True by default
best_model = grid_dt.best_estimator_

# Evaluate test set accuracy
test_acc = best_model.score(X_test,y_test)

# Print test set accuracy
print("Test set accuracy of best model: {:.3f}".format(test_acc))
```

roc-auc:
```
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)

grid_dt.fit(X_train, y_train)

# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test,y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
```

neg_mean_squared_error, Random Forest:
```
# Basic imports
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
# Define a grid of hyperparameter 'params_rf'
params_rf = {
'n_estimators': [300, 400, 500],
'max_depth': [4, 6, 8],
'min_samples_leaf': [0.1, 0.2],
'max_features': ['log2', 'sqrt']
}
# Instantiate 'grid_rf'
# verbose controls verbosity. higher the value, more messages printed during fitting
grid_rf = GridSearchCV(estimator=rf,
param_grid=params_rf,
cv=3,
scoring='neg_mean_squared_error',
verbose=1,
n_jobs=-1)

# Fit 'grid_rf' to the training set
grid_rf.fit(X_train, y_train)

# Extract best hyperparameters from 'grid_rf'
best_hyperparams = grid_rf.best_params_
print('Best hyerparameters:\n', best_hyperparams)

# Extract best model from 'grid_rf'
best_model = grid_rf.best_estimator_
# Predict the test set labels
y_pred = best_model.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```

## Introduction to data visualization with matplotlib

basic:
```
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()
```

### Customization
- Adding and Choosing markers
- Setting the linestyle
- Eliminating lines with linestyle
- Choosing color
- Customizing the axes labels
- Setting the axis label
- Adding a title


```
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"],
seattle_weather["MLY-TAVG-NORMAL"],
marker="v", linestyle="--", color="r")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Average temperature (Fahrenheit degrees)")
ax.set_title("Weather in Seattle")
plt.show()
```

### Small multiples with plt.subplots
2-d axes:
```
fig, ax = plt.subplots(3, 2)
plt.show()

# an array of Axes objects with a shape of 3 by 2
ax.shape

# have to index into this object and call the plot method on an element of the array
ax[0, 0].plot(seattle_weather["MONTH"],
seattle_weather["MLY-PRCP-NORMAL"],
color='b')
plt.show()
```

1-d axes:
```
fig, ax = plt.subplots(2, 1)
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"],
color='b')
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-25PCTL"],
linestyle='--', color='b')
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-75PCTL"],
linestyle='--', color='b')
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"],
color='r')
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-25PCTL"],
linestyle='--', color='r')
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-75PCTL"],
linestyle='--', color='r')
ax[0].set_ylabel("Precipitation (inches)")
ax[1].set_ylabel("Precipitation (inches)")
ax[1].set_xlabel("Time (months)")
plt.show()
```

### Sharing the y-axis range
```
fig, ax = plt.subplots(2, 1, sharey=True)
```

### Plotting time-series data
```
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()
```

### Zooming in on a decade
```
# slicing date
sixties = climate_change["1960-01-01":"1969-12-31"]
fig, ax = plt.subplots()
ax.plot(sixties.index, sixties['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()
```

### Zooming in on one year
```
sixty_nine = climate_change["1969-01-01":"1969-12-31"]
fig, ax = plt.subplots()
ax.plot(sixty_nine.index, sixty_nine['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()
```

### Plotting two time-series together
```
import pandas as pd
climate_change = pd.read_csv('climate_change.csv',
parse_dates=["date"],
index_col="date")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change["co2"])
ax.plot(climate_change.index, climate_change["relative_temp"])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm) / Relative temperature')
plt.show()
```

### Using twin axes
```
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change["co2"],
color='blue')
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)', color='blue')
ax.tick_params('y', colors='blue')

ax2 = ax.twinx()
ax2.plot(climate_change.index,
climate_change["relative_temp"],
color='red')
ax2.set_ylabel('Relative temperature (Celsius)',
color='red')
ax2.tick_params('y', colors='red')
plt.show()
```

### A function that plots time-series
```
def plot_timeseries(axes, x, y, color, xlabel, ylabel):
axes.plot(x, y, color=color)
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel, color=color)
axes.tick_params('y', colors=color)
```

using function:
```
fig, ax = plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['co2'],
'blue', 'Time', 'CO2 (ppm)')
ax2 = ax.twinx()
plot_timeseries(ax, climate_change.index,
climate_change['relative_temp'],
'red', 'Time', 'Relative temperature (Celsius)')
plt.show()
```

### Annotating timeseries data
- Annotation
- Positioning the text
- Adding arrows to annotation

```
fig, ax = plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['co2'],
'blue', 'Time', 'CO2 (ppm)')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index,
climate_change['relative_temp'],
'red', 'Time', 'Relative temperature (Celsius)')
# xy is the x and y values respectively
# arrowprops takes in dictionary defines the property of arrow. if empty dict, arrow will have default property

ax2.annotate(">1 degree",
xy=(pd.Timestamp('2015-10-06'), 1),
xytext=(pd.Timestamp('2008-10-06'), -0.2),
arrowprops={"arrowstyle":"->", "color":"gray"})

plt.show()
```

### stacked bar graph
```
fig, ax = plt.subplots
ax.bar(medals.index, medals["Gold"], label="Gold")
ax.bar(medals.index, medals["Silver"], bottom=medals["Gold"],
label="Silver")
ax.bar(medals.index, medals["Bronze"],
bottom=medals["Gold"] + medals["Silver"],
label="Bronze")
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel("Number of medals")
ax.legend()
plt.show()
```

### histograms
```
fig, ax = plt.subplots()
ax.hist(mens_rowing["Height"], label="Rowing", bins=5)
ax.hist(mens_gymnastic["Height"], label="Gymnastics", bins=5)
ax.set_xlabel("Height (cm)")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()
```

setting bin boundaries:
```
fig, ax = plt.subplots()
ax.hist(mens_rowing["Height"], label="Rowing",
bins=[150, 160, 170, 180, 190, 200, 210])
ax.hist(mens_gymnastic["Height"], label="Gymnastics",
bins=[150, 160, 170, 180, 190, 200, 210])
ax.set_xlabel("Height (cm)")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()
```

transparency:
```
Customizing histograms: transparency
ax.hist(mens_rowing["Height"], label="Rowing",
bins=[150, 160, 170, 180, 190, 200, 210],
histtype="step")
ax.hist(mens_gymnastic["Height"], label="Gymnastics",
bins=[150, 160, 170, 180, 190, 200, 210],
histtype="step")
ax.set_xlabel("Height (cm)")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()
```

### adding error bars
summarizes distribution of data into one number such as standard deviation

yerr: takes in additional number and displays as additional vertical marker

to barplot:
```
fig, ax = plt.subplots()
# "Rowing" is the naming for the xaxis
ax.bar("Rowing",
mens_rowing["Height"].mean(),
yerr=mens_rowing["Height"].std())
ax.bar("Gymnastics",
mens_gymnastics["Height"].mean(),
yerr=mens_gymnastics["Height"].std())
ax.set_ylabel("Height (cm)")
plt.show()
```

to line plot:
```
fig, ax = plt.subplots()
ax.errorbar(seattle_weather["MONTH"],
seattle_weather["MLY-TAVG-NORMAL"],
yerr=seattle_weather["MLY-TAVG-STDDEV"])
ax.errorbar(austin_weather["MONTH"],
austin_weather["MLY-TAVG-NORMAL"],
yerr=austin_weather["MLY-TAVG-STDDEV"])
ax.set_ylabel("Temperature (Fahrenheit)")
plt.show()
```

### boxplot
outlier: outside of roughly 99% of distribution if data is Gaussian or normal
```
fig, ax = plt.subplots()
# take note of bracket inside boxplot()
ax.boxplot([mens_rowing["Height"],
mens_gymnastics["Height"]])
ax.set_xticklabels(["Rowing", "Gymnastics"])
ax.set_ylabel("Height (cm)")
plt.show()
```

### scatterplot
bi-variate comparison. different variables comparison for same observation.

```
fig, ax = plt.subplots()
ax.scatter(climate_change["co2"], climate_change["relative_temp"])
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()
```

### two scatterplot in same axis
```
eighties = climate_change["1980-01-01":"1989-12-31"]
nineties = climate_change["1990-01-01":"1999-12-31"]
fig, ax = plt.subplots()
ax.scatter(eighties["co2"], eighties["relative_temp"],
color="red", label="eighties")
ax.scatter(nineties["co2"], nineties["relative_temp"],
color="blue", label="nineties")
ax.legend()
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()
```

Encoding a third variable by color:
```
fig, ax = plt.subplots()
ax.scatter(climate_change["co2"], climate_change["relative_temp"],
c=climate_change.index)
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()
```

### Choosing a style
The available styles:
h ps://matplotlib.org/gallery/style_sheets/style_sheets_refere

```
plt.style.use("ggplot")
fig, ax = plt.subplots()
```

Back to the default
```
plt.style.use("default")
```

Guidelines for choosing plotting style:
- Dark backgrounds are usually less visible
- If color is important, consider choosing colorblind-friendly
options "seaborn-colorblind" or "tableau-colorblind10"
- If you think that someone will want to print your figure, use
less ink
- If it will be printed in black-and-white, use the "grayscale"
style


### Saving the figure to file
```
fig, ax = plt.subplots()
ax.bar(medals.index, medals["Gold"])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel("Number of medals")
# fig will not show without .show
fig.savefig("gold_medals.png")

ls
```
gold_medals.png

Different file formats:
jpg takes less space compared to png
```
fig.savefig("gold_medals.jpg")
# compression for jpg. higher value less compresion
fig.savefig("gold_medals.jpg", quality=50)
# vector file. can edit in gimp or adobe
fig.savefig("gold_medals.svg")
```

Resolution:
```
fig.savefig("gold_medals.png", dpi=300)
```

Size:
```
fig.set_size_inches([5, 3])
```

### Getting unique values of a column in series
```
sports = summer_2016_medals["Sport"].unique()
print(sports)
['Rowing' 'Taekwondo' 'Handball' 'Wrestling'
'Gymnastics' 'Swimming' 'Basketball' 'Boxing'
'Volleyball' 'Athletics']
```

### Bar-chart loop through series
Bar-chart of heights for all sports
```
fig, ax = plt.subplots()
for sport in sports:
sport_df = summer_2016_medals[summer_2016_medals["Sport"] == sport]
ax.bar(sport, sport_df["Height"].mean(),
yerr=sport_df["Height"].std())
ax.set_ylabel("Height (cm)")
ax.set_xticklabels(sports, rotation=90)
plt.show()
```

### Matplotlib gallery
https://matplotlib.org/gallery.html

examples:
- Visualizing images with pseudo-color https://matplotlib.org/users/image_tutorial.html
- Animations https://matplotlib.org/api/animation_api.html
- geospatial data https://scitools.org.uk/cartopy/docs/latest/
- Pandas + Matplotlib = Seaborn

### Seaborn example gallery
https://seaborn.pydata.org/examples/index.html

## Introduction to data visualization with Seaborn
Seaborn is a Python data visualization library. works well with pandas data structure

### relational plots
Show the relationship between two quantitative variables
Two types of relational plots: scatter plots and line plots
- Scatter plots: Each plot point is an independent
observation
- Line plots: Each plot point represents the same "thing", typically tracked over time
- Can use relplot()

#### scatterplot
```
import seaborn as sns
import matplotlib.pyplot as plt
height = [62, 64, 69, 75, 66,
68, 65, 71, 76, 73]
weight = [120, 136, 148, 175, 137,
165, 154, 172, 200, 187]
sns.scatterplot(x=height, y=weight)
plt.show()
```

with hue:
```
sns.scatterplot(x="total_bill",
y="tip",
data=tips,
hue="smoker")
plt.show()
```

setting hue order:
```
sns.scatterplot(x="total_bill",
y="tip",
data=tips,
hue="smoker",
hue_order=["Yes",
"No"])
plt.show()
```

Specifying hue colors
```
hue_colors = {"Yes": "black",
"No": "red"}
sns.scatterplot(x="total_bill",
y="tip",
data=tips,
hue="smoker",
palette=hue_colors)
plt.show()
```

Using HTML hex color codes with hue:
```
hue_colors = {"Yes": "#808080",
"No": "#00FF00"}
```

#### scatterplot() vs. relplot()
can use relplot for relational plots

Using scatterplot()
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x="total_bill",
y="tip",
data=tips)
plt.show()
```

using relplot:
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(x="total_bill",
y="tip",
data=tips,
kind="scatter")
plt.show()
```

#### Relplot() Subplots in rows and columns
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(x="total_bill",
y="tip",
data=tips,
kind="scatter",
col="smoker",
row="time")
plt.show()
```

Wrapping and ordering columns
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(x="total_bill",
y="tip",
data=tips,
kind="scatter",
col="day",
col_wrap=2,
col_order=["Thur","Fri","Sat","Sun"])
plt.show()
```

#### more customization
Point size:
```
sns.relplot(x="total_bill",
y="tip",
data=tips,
kind="scatter",
size="size")
```

Point size and hue:
```
sns.relplot(x="total_bill",
y="tip",
data=tips,
kind="scatter",
size="size",
hue="size")
```

Point style:
```
sns.relplot(x="total_bill",
y="tip",
data=tips,
kind="scatter",
hue="smoker",
style="smoker")
```

transparency:
```
# Set alpha to be between 0 and 1
sns.relplot(x="total_bill",
y="tip",
data=tips,
kind="scatter",
alpha=0.4)
```

marker:
```
sns.relplot(x="hour", y="NO_2_mean",
data=air_df_loc_mean,
kind="line",
style="location",
hue="location",
markers=True)
```

turning off line style:
```
sns.relplot(x="hour", y="NO_2_mean",
data=air_df_loc_mean,
kind="line",
style="location",
hue="location",
markers=True,
dashes=False)
```

#### lineplot
```
import matplotlib.pyplot as plt
import seaborn as sns
sns.relplot(x="hour", y="NO_2_mean",
data=air_df_mean,
kind="line")
plt.show()
```

if line plot given multiple observations per x-value, it would aggregate into single summary measure.
default it would use mean:
(note: would automatically display confidence interval. Assumes dataset is a random sample 95% confident that the mean is within this interval)
```
import matplotlib.pyplot as plt
import seaborn as sns
sns.relplot(x="hour", y="NO_2",
data=air_df,
kind="line")
plt.show()
```

Replacing confidence interval with standard deviation
```
sns.relplot(x="hour", y="NO_2",
data=air_df,
kind="line",
ci="sd")
```

turn off confidence interval:
```
sns.relplot(x="hour", y="NO_2",
data=air_df,
kind="line",
ci=None)
```


### Categorical plots
Show the distribution of a quantitative variable within categories de ned by a categorical
variable
- Examples: count plots, bar plots, boxplots, point plots
- Involve a categorical variable
- Comparisons between groups
- Can use catplot()

#### countplot
list/series:
```
import seaborn as sns
import matplotlib.pyplot as plt
gender = ["Female", "Female",
"Female", "Female",
"Male", "Male", "Male",
"Male", "Male", "Male"]
sns.countplot(x=gender)
plt.show()
```

dataframe:
```
# Import Matplotlib, Pandas, and Seaborn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Create a DataFrame from csv file
df=pd.read_csv(csv_filepath)

# Create a count plot with "Spiders" on the x-axis
sns.countplot(x='Spiders',data=df)

# Display the plot
plt.show()
```

with hue:
```
sns.countplot(x="smoker",
data=tips,
hue="sex")
```

#### countplot() vs. catplot()
```
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x="how_masculine",
data=masculinity_data)
plt.show()
```

catplot():
for categorical plots
same advantage as relplot()
```
import matplotlib.pyplot as plt
import seaborn as sns
sns.catplot(x="how_masculine",
data=masculinity_data,
kind="count")
plt.show()
```

#### catplot()

horizontal:
```
sns.catplot(y="Internet usage", data=survey_data,
            kind="count")
```

Changing the order:
```
import matplotlib.pyplot as plt
import seaborn as sns
category_order = ["No answer", "Not at all", "Not very", "Somewhat","Very"]
sns.catplot(x="how_masculine",
data=masculinity_data,
kind="count",
order=category_order)
plt.show()
```

barplot:
automatically show 95% ci.
Assuming our data is a random sample of some population, we can be 95% sure that the true population mean in each group lies within the confidence interval shown.
When the y-variable is True/False, bar plots will show the percentage of responses reporting True.
```
sns.catplot(x="day",
y="total_bill",
data=tips,
kind="bar")
```

remove ci:
```
sns.catplot(x="day",
y="total_bill",
data=tips,
kind="bar",
ci=None)
```

subplot:
```
# Create column subplots based on age category
sns.catplot(y="Internet usage", data=survey_data,
            kind="count",col='Age Category')

# Show plot
plt.show()
```

#### boxplot
- Shows the distribution of quantitative data
- See median, spread, skewness, and outliers
- Facilitates comparisons between groups

seaborn have boxplot() function but use catplot so easier to create subplots
```
g = sns.catplot(x="time",
y="total_bill",
data=tips,
kind="box")
plt.show()
```

Change the order of categories
```
g = sns.catplot(x="time",
y="total_bill",
data=tips,
kind="box",
order=["Dinner",
"Lunch"])
```

Omitting the outliers using `sym`
```
g = sns.catplot(x="time",
y="total_bill",
data=tips,
kind="box",
sym="")
```

Changing the whiskers using `whis`:
By default, the whiskers extend to 1.5 * the interquartile range
Make them extend to 2.0 * IQR: whis=2.0
Show the 5th and 95th percentiles: whis=[5, 95]
Show min and max values: whis=[0, 100]

Add subgroups so each box plot is colored:
```
# Create a box plot with subgroups and omit the outliers
sns.catplot(x='internet',y='G3',data=student_data,kind='box',sym='',hue='location')
```

#### pointplot
- Points show mean of quantitative variable
- Vertical lines show 95% confidence intervals

```
sns.catplot(x="age",
y="masculinity_important",
data=masculinity_data,
hue="feel_masculine",
kind="point")
plt.show()
```

Disconnecting the points:
```
sns.catplot(x="age",
y="masculinity_important",
data=masculinity_data,
hue="feel_masculine",
kind="point",
join=False)
plt.show()
```

Displaying the median:
median more robust to outliers
```
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import median
sns.catplot(x="smoker",
y="total_bill",
data=tips,
kind="point",
estimator=median)
plt.show()
```

Add caps to confidence intervals:
```
import matplotlib.pyplot as plt
import seaborn as sns
sns.catplot(x="smoker",
y="total_bill",
data=tips,
kind="point",
capsize=0.2)
plt.show()
```

Turning off confidence intervals:
```
sns.catplot(x="smoker",
y="total_bill",
data=tips,
kind="point",
ci=None)
```

#### Point plots vs. line plots
Both show:
-Mean of quantitative variable
- 95% confidence intervals for the mean

Differences:
- Line plot has quantitative variable (usually time) on x-axis
- Point plot has categorical variable on x-axis

#### Point plots vs. bar plots
Both show:
- Mean of quantitative variable
- 95% confidence intervals for the mean

But in point plots, easier to compare heights of the subgroup points since they are stacked above each other. this is easier than comparing heights in bar plots

### Changing the figure style
Figure "style" includes background and axes
Preset options: "white", "dark", "whitegrid", "darkgrid", "ticks"
Default is white
```
# add before plot
sns.set_style()
```

### Changing the palette
Figure "palette" changes the color of the main elements of the plot

```
# add before plot
sns.set_palette()
```
Use preset palettes or create a custom palette

Diverging palettes: 
- RdBu
- PRGn
- RdBu_r
- PRGn_r

Sequential palettes:
- Greys
- Blues
- PuRd
- GnBu

Custom palettes:
```
custom_palette = ["red", "green", "orange", "blue",
"yellow", "purple"]
sns.set_palette(custom_palette)
```

```
custom_palette = ['#FBB4AE', '#B3CDE3', '#CCEBC5',
'#DECBE4', '#FED9A6', '#FFFFCC',
'#E5D8BD', '#FDDAEC', '#F2F2F2']
sns.set_palette(custom_palette)
```

### Changing the scale
Figure "context" changes the scale of the plot elements and labels
```
sns.set_context()
```
Smallest to largest: "paper", "notebook", "talk", "poster".
Default is paper.

### FacetGrid vs. AxesSubplot objects
Seaborn plots create two different types of objects: FacetGrid and AxesSubplot

```
# to see which type is the plot
g = sns.scatterplot(x="height", y="weight", data=df)
type(g)
```

FacetGrid consists of one or more AxeSubplots.

FacetGrid: 
- Eg relplot() , catplot().
- Can create subplots

AxesSubplot:
- Eg scatterplot() , countplot() , etc. 
- Only creates a single plot

### Adding a title 

```
g = sns.boxplot(x="Region",
y="Birthrate",
data=gdp_data)
```

to FacetGrid
```
g.fig.suptitle("New Title")
```

to AxesSubplot
```
g.set_title("New Title",
y=1.03)
```

for subplots:
```
g.fig.suptitle("New Title",
y=1.03)
g.set_titles("This is {col_name}")
```

### Adjusting height of title 

default y is 1.
same for facetgrid and axesubplot
```
g = sns.catplot(x="Region",
y="Birthrate",
data=gdp_data,
kind="box")
g.fig.suptitle("New Title",
y=1.03)
plt.show()
```

### Adding axis labels
same for facetgrid and axesubplot
```
g = sns.catplot(x="Region",
y="Birthrate",
data=gdp_data,
kind="box")

g.set(xlabel="New X Label",
ylabel="New Y Label")
plt.show()
```

### Rotating x-axis tick labels
use matplotlib
```
g = sns.catplot(x="Region",
y="Birthrate",
data=gdp_data,
kind="box")
plt.xticks(rotation=90)
```