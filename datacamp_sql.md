# SQL

### Introduction to SQL
SQL, which stands for Structured Query Language, is a language for interacting with data stored in something called a relational database.
Each row, or record, of a table contains information about a single entity. Each column contains single attribute for all rows in the table.

 A query is a request for data from a database table (or combination of tables).
 
 In SQL, keywords are not case-sensitive. That said, it's good practice to make SQL keywords uppercase to distinguish them from other parts of your query, like column and table names.

It's also good practice (but not necessary for the exercises in this course) to include a semicolon at the end of your query. 

To select multiple columns from a table, simply separate the column names with commas!

select all columns: SELECT *

If you only want to return a certain number of results, you can use the LIMIT keyword to limit the number of rows returned:
SELECT *
FROM people
LIMIT 10;

If you want to select all the unique values from a column
SELECT DISTINCT language
FROM films;

The COUNT statement lets you do this by returning the number of rows in one or more columns. Count dont include null.

SELECT COUNT(*)
FROM people;

It's also common to combine COUNT with DISTINCT to count the number of distinct values in a column:
SELECT COUNT(DISTINCT birthdate)
FROM people;

WHERE keyword allows you to filter based on both text and numeric values in a table. There are a few different comparison operators you can use:
= equal
<> not equal. use <> and not != for the not equal operator, as per the SQL standard.
< less than
> greater than
<= less than or equal to
>= greater than or equal to

SELECT title
FROM films
WHERE title = 'Metropolis';
Notice that the WHERE clause always comes after the FROM statement!

Important: in PostgreSQL (the version of SQL we're using), you must use single quotes with WHERE

You can build up your WHERE queries by combining multiple conditions with the AND keyword.
Note that you need to specify the column name separately for every AND condition

BETWEEN keyword provides a useful shorthand for filtering values within a specified range. 
It's important to remember that BETWEEN is inclusive, meaning the beginning and end values are included in the results!

The IN operator allows you to specify multiple values in a WHERE clause, making it easier and quicker to specify multiple OR conditions!
SELECT name
FROM kids
WHERE age IN (2, 4, 6, 8, 10);

IS NULL represents a missing or unknown value. 
opposite is IS NOT NULL

In SQL, the LIKE operator can be used in a WHERE clause to search for a pattern in a column. 
You can also use the NOT LIKE operator to find records that don't match the pattern you specify.
The % wildcard will match zero, one, or many characters in text. 
The _ wildcard will match a single character. 

SELECT name
FROM companies
WHERE name LIKE 'Data%';

SQL provides a few functions, called aggregate functions,
eg SELECT AVG(budget)

SQL assumes that if you divide an integer by an integer, you want to get an integer back. So be careful when dividing!
If you want more precision when dividing, you can add decimal places to your numbers. For example,
SELECT (4.0 / 3.0) AS result;
gives you the result you would expect: 1.333.

Aliasing simply means you assign a temporary name to something. To alias, you use the AS keyword
SELECT MAX(budget) AS max_budget,
       MAX(duration) AS max_duration
FROM films;

The ORDER BY keyword is used to sort results in ascending or descending order according to the values of one or more columns.
By default ORDER BY will sort in ascending order. If you want to sort the results in descending order, you can use the DESC keyword
SELECT title
FROM films
ORDER BY release_year DESC;

ORDER BY can also be used to sort on multiple columns. It will sort by the first column specified, then sort by the next, then the next, and so on.
SELECT birthdate, name
FROM people
ORDER BY birthdate, name;

GROUP BY always goes after the FROM clause!
SQL will return an error if you try to SELECT a field that is not in your GROUP BY clause without using it to calculate some kind of value about the entire group.
GROUP BY allows you to group a result by one or more columns, like so:
SELECT sex, count(*)
FROM employees
GROUP BY sex
ORDER BY count DESC;

aggregate functions can't be used in WHERE clauses. 
hat's where the HAVING clause comes in. For example,
SELECT release_year
FROM films
GROUP BY release_year
HAVING COUNT(title) > 10
ORDER BY release_year;

### Joining Data in SQL
