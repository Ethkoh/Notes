# Python Notes

For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

renaming some cols
df.rename(columns = {'Revenue (Millions)':'Rev_M','Runtime (Minutes)':'Runtime_min'},inplace=True)

create a column with basic arithmetic
df['AvgRating'] = (df['Rating'] + df['Metascore']/10)/2

apple
The general structure is:
You define a function that will take the column values you want to play with to come up with your logic. Here the only two columns we end up using are genre and rating.
You use an apply function with lambda along the row with axis=1. The general syntax is: df.apply(lambda x: func(x['col1'],x['col2']),axis=1)

def custom_rating(genre,rating):
    if 'Thriller' in genre:
        return min(10,rating+1)
    elif 'Comedy' in genre:
        return max(0,rating-1)
    else:
        return rating
        
df['CustomRating'] = df.apply(lambda x: custom_rating(x['Genre'],x['Rating']),axis=1)

Filtering dataframe:
Single condition: dataframe with all movies rated greater than 8
df_gt_8 = df[df['Rating']>8]

Multiple conditions: AND - dataframe with all movies rated greater than 8 and having more than 100000 votes
And_df = df[(df['Rating']>8) & (df['Votes']>100000)]

Multiple conditions: OR - dataframe with all movies rated greater than 8 or having a metascore more than 90
Or_df = df[(df['Rating']>8) | (df['Metascore']>80)]

Multiple conditions: NOT - dataframe with all emovies rated greater than 8 or having a metascore more than 90 have to be excluded
Not_df = df[~((df['Rating']>8) | (df['Metascore']>80))]

Filter with split
create a new column
df['num_words_title'] = df.apply(lambda x : len(x['Title'].split(" ")),axis=1)
simple filter on new column
new_df = df[df['num_words_title']>=4]
new_df.head()

OR new_df = df[df.apply(lambda x : len(x['Title'].split(" "))>=4,axis=1)]

change column type. Axis =1 to apply function only to one column which in this case is price. 
df['Price'] is to replace onto the dataframe.
df['Price'] = df.apply(lambda x: int(x['Price'].replace(',', '')),axis=1)

The map() method only works on panda series where different types of operation can be applied to the items in the series. Dont work for 2 columns
The apply () method works on panda series and data frames with a variety of functions easily applied depending on the datatype.
The applymap() method works on the entire pandas data frame where the input function is applied to every element individually. 

## impute data with groupby mean
- df.groupby simply groups the dataframe into sub-dataframes (groups), such that each group only contains one Brand
- transform() will apply a function to a dataframe - so to each of the individual groups created in groupby
- the nameless function (a lambda function) calls the DataFrame's fillna() method on each dataframe, using just the mean() to fill the gaps
```
df["Value"] = df.groupby("Brand")["Value"].transform(lambda x: x.fillna(x.mean()))
```