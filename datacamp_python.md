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
rain and test sets are vital to ensure that your supervised learning model is able to generalize well to new data. 
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
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()
hidden_layer_values = np.array([node_0_value, node_1_value]
print(hidden_layer_values)
[5, 1]
output = (hidden_layer_values * weights['output']).sum()
print(output)
9
```