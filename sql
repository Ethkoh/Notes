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

The COUNT statement lets you do this by returning the number of rows in one or more columns.
SELECT COUNT(*)
FROM people;
