# DataCamp SQL

## Introduction to SQL
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

### DISTINCT
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

### IN operator
 allows you to specify multiple values in a WHERE clause, making it easier and quicker to specify multiple OR conditions!
SELECT name
FROM kids
WHERE age IN (2, 4, 6, 8, 10);

IS NULL represents a missing or unknown value. 
opposite is IS NOT NULL

In SQL, the LIKE operator can be used in a WHERE clause to search for a pattern in a column. 
You can also use the NOT LIKE operator to find records that don't match the pattern you specify.

### Wildcard % and _
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

## Joining Data in SQL

can join table with itself

if same name in other database join, will have error if never specify which database SELECT columns is from
```
SELECT *
FROM left_table
INNER JOIN right_table
ON left_table.id = right_table.id;

SELECT c1.name AS city, c2.name AS country
FROM cities AS c1
INNER JOIN countries AS c2
ON c1.country_code = c2.code;
```
When joining tables with a common field name, You can use USING as a shortcut instead of ON. rmbr to use ():
```
SELECT *
FROM countries
  INNER JOIN economies
    USING(code)
```

CASE WHEN THEN can do if-else for sql
You can use CASE with WHEN, THEN, ELSE, and END to define a new grouping field.
```
SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus       
FROM populations
WHERE year = 2015;
```

Use INTO to save the result of the previous query
Use INTO before FROM

3 types of outer joins: left, right, full

comments use /* comments */

### Left Join, Group BY
```
-- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c)
FROM countries AS c
  -- Left join with economies (alias as e)
  LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year=2010
-- Group by region
GROUP BY region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC;
```

### 2 Left Join
```
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
  LEFT JOIN countries
    ON cities.country_code = countries.code
  LEFT JOIN languages
    ON countries.code = languages.code
ORDER BY city, language;
```

### Full Join
```
SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM currencies
  -- 4. Join to currencies
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;
```

```
SELECT c1.name AS country, region, l.name AS language,
       frac_unit, 	basic_unit
-- 1. From countries (alias as c1)
FROM countries AS c1
  -- 2. Join with languages (alias as l)
  FULL JOIN languages AS l
    -- 3. Match on code
    USING (code)
  -- 4. Join with currencies (alias as c2)
  FULL JOIN currencies AS c2
    -- 5. Match on code
    USING (code)
-- 6. Where region like Melanesia and Micronesia
WHERE region LIKE 'M%esia';
```

### Cross Join
number of ids will be num of ids 1 * ids 2
no need on/using clause

### Union all vs union
union all include all duplicates. they dont lookup. simply stack one table on top of another.
union dont include duplciates. each entries 1

### Intersect
only records common in both tables
if more than one column, look for record same 

### Except
include data only in one table

### Semi Join / Subqueries
chooses records in the first table where conditions are met in the second table 

example:
```
-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
-- Order by name
ORDER BY name;
```
Sometimes problems solved with semi-joins can also be solved using an inner join.
```
SELECT DISTINCT languages.name AS language
FROM languages
INNER JOIN countries
ON languages.code = countries.code
WHERE region = 'Middle East'
ORDER BY language;
```

### Anti Join
chooses records in the first table where conditions are not met in the second table

example:
```
-- 3. Select fields
SELECT c1.code,c1.name
  -- 4. From Countries
  FROM countries AS c1
  -- 5. Where continent is Oceania
  WHERE continent ='Oceania'
  	-- 1. And code not in
  	AND code NOT IN
  	-- 2. Subquery
  	(SELECT code
  	FROM currencies
);
```

### summary of joins
1. inner
- self
2. outer
- left
- right
- full
3. cross join
4. semi join
5. anti join

### summary of set theory
1. union 
2. union all
3. intercept (not same as inner join)
4. except

### Example of joins and set theories
```
-- Select the city name
SELECT name
  -- Alias the table where city name resides
  FROM cities AS c1
  -- Choose only records matching the result of multiple set theory clauses
  WHERE country_code IN
(
    -- Select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- Get all additional (unique) values of the field from currencies AS c2  
    UNION ALL
    SELECT c2.code
    FROM currencies AS c2
    -- Exclude those appearing in populations AS p
    EXCEPT
    SELECT p.country_code
    FROM populations AS p
);
```

### Subquery in SELECT
= nested query

commonly Subqueries inside WHERE and SELECT clauses

Example:
```
/*SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;
*/
```
Alternative solution:
```
SELECT name AS country,
  (SELECT COUNT(name)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries 
ORDER BY cities_num DESC, country
LIMIT 9;
```
### Subquery in WHERE
Most common where subquery found
example:
```
SELECT name, fert_rate
FROM states
WHERE continent = 'Asia'
AND fert_rate <
(SELECT AVG(fert_rate)
FROM states);
```
another example:
```
-- Select fields
SELECT code,inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
 WHERE year = 2015 AND code NOT IN 
  	-- Subquery
  	(SELECT code
  	 FROM countries
  	 WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%')) 
-- Order by inflation rate
ORDER BY inflation_rate;
```

 
### Subquery in FROM 
as a temporary table

example:
```
SELECT DISTINCT monarchs.continent, subquery.max_perc
FROM monarchs,
(SELECT continent, MAX(women_parli_perc) AS max_perc
FROM states
GROUP BY continent) AS subquery
WHERE monarchs.continent = subquery.continent
ORDER BY continent;
```

