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
file 

### assign excel file to variable
d



