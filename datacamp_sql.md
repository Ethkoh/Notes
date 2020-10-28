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

## Introduction to Relational Databases in SQL

advantage of splitting tables is to reduce redundancy

these help preserve data quality:
1. constraints
2. keys
3. referential integrity

### metadatabases
holds information regarding your databases

### create tables
semicolon at the end impt
```
CREATE TABLE table_name (
 column_a data_type,
 column_b data_type,
 column_c data_type
);
```

### add columns to table
add columns you can use the following SQL query:
```
ALTER TABLE table_name
ADD COLUMN column_name data_type;
```

### INSERT DISTINCT records INTO the new tables
```
INSERT INTO organizations
SELECT DISTINCT organization,
organization_sector
FROM university_professors;
```

### manually INSERT INTO statement for values
```
INSERT INTO table_name (column_a, column_b)
VALUES ("value_a", "value_b");
```

### rename column names
```
ALTER TABLE table_name
RENAME COLUMN old_name TO new_name;
```

### drop column in table
```
ALTER TABLE table_name
DROP COLUMN column_name;
```

### delete table
```
DROP TABLE table_name;
```
### integrity constraints
Why constraints?
Constraints give the data structure
Constraints help with consistency, and thus data quality
Data quality is a business advantage / data science prerequisite
Enforcing is di cult, but PostgreSQL helps
1. Attribute constraints, e.g. data types on columns (Chapter 2)
2. Key constraints, e.g. primary keys (Chapter 3)
3. Referential integrity constraints, enforced through foreign keys (Chapter 4)

### casting
on the fly datatype conversion
```
SELECT CAST(some_column AS integer)
FROM table;
```
```
SELECT temperature * CAST(wind_speed AS integer) AS wind_chill
FROM weather;
```

### types of database constraints
- Foreign keys are special constraints on attributes that act as links to other database tables.
- A data type is a simple form of an attribute constraint, leading to a consistent data type across a database column.
- Primary keys are special constraints on attributes that uniquely identify each record in a table.

### working with data types
- Enforced on columns (i.e. attributes)
- Define the so-called "domain" of a column
- Define what operations are possible
- Enfore consistent storage of values

### common datatype for postgreSQL
1. text : character strings of any length
2. varchar [ (x) ] : a maximum of n characters
3. char [ (x) ] : a  xed-length string of n characters
4. boolean : can only take three states, e.g. TRUE , FALSE and NULL (unknown)
5. date , time and timestamp : various formats for date and time calculations
6. numeric : arbitrary precision numbers, e.g. 3.1457
7. integer : whole numbers in the range of -2147483648 and +2147483647
8. bigint: for bigger number of integers
9. serial: set incremental number

### Alter types after table creation
```
ALTER TABLE students
ALTER COLUMN name
TYPE varchar(128);
```
```
ALTER TABLE students
ALTER COLUMN average_grade
TYPE integer
-- Turns 5.54 into 6, not 5, before type conversion
USING ROUND(average_grade);
```

### Convert types USING a function
If you don't want to reserve too much space for a certain varchar column, you can truncate the values before converting its type
```
ALTER TABLE table_name
ALTER COLUMN column_name
TYPE varchar(x)
USING SUBSTRING(column_name FROM 1 FOR x)
```
You should read it like this: Because you want to reserve only x characters for column_name, you have to retain a SUBSTRING of every value, i.e. the first x characters of it, and throw away the rest. This way, the values will fit the varchar(x) requirement. However, it's best not to truncate any values in your database.

### The not-null constraint
- Disallow NULL values in a certain column
- Must hold true for the current state
- Must hold true for any future state

### add not-null when creating table
```
CREATE TABLE students (
ssn integer not null,
lastname varchar(64) not null,
home_phone integer,
office_phone integer
);
```

### add not-null after table created
```
ALTER TABLE students
ALTER COLUMN home_phone
SET NOT NULL;
```

### The unique constraint
important before making columns primary key
- Disallow duplicate values in a column
- Must hold true for the current state
- Must hold true for any future state

### Adding unique constraints when creating table
```
CREATE TABLE table_name (
column_name UNIQUE
);
```

### Adding unique constraints after table created
Note that this is different from the ALTER COLUMN syntax for the not-null constraint. Also, you have to give the constraint a name some_name.
```
ALTER TABLE table_name
ADD CONSTRAINT some_name UNIQUE(column_name);
```

### What is a key?
Attribute(s) that identify a record uniquely
As long as attributes can be removed: superkey
If no more attributes can be removed: minimal superkey or key

### how to find key?
1. Count the distinct records for all possible combinations of columns. If the resulting number x equals the number of all rows in the table for a combination, you have discovered a superkey.

2. Then remove one column after another until you can no longer remove columns without seeing the number x decrease. If that is the case, you have discovered a (candidate) key.

### primary key
- One primary key per database table, chosen from candidate keys
- Uniquely identifies records, e.g. for referencing in other tables
- Unique and not-null constraints both apply
- Primary keys are time-invariant. must hold for current and future data

example 1:
```
CREATE TABLE products (
product_no integer UNIQUE NOT NULL,
name text,
price numeric
);
```
example 2: more specific
```
CREATE TABLE products (
product_no integer PRIMARY KEY,
name text,
price numeric
);
```
example 3: more than 1 primary key
```
CREATE TABLE example (
a integer,
b integer,
c integer,
PRIMARY KEY (a, c)
);
```
example 4: alter existing table 
```
ALTER TABLE table_name
ADD CONSTRAINT some_name PRIMARY KEY (column_name)
```

### Surrogate keys
created artificial primary key
- Primary keys should be built from as few columns as possible
- Primary keys should never change over time

Adding a surrogate key with serial data type
```
ALTER TABLE cars
ADD COLUMN id serial PRIMARY KEY;
INSERT INTO cars
VALUES ('Volkswagen', 'Blitz', 'black');
```

### create primary key with 2 columns
```
-- Count the number of distinct rows with columns make, model
SELECT COUNT(DISTINCT(make, model)) 
FROM cars;

-- Add the id column
ALTER TABLE cars
ADD COLUMN id varchar(128);

-- Update id with make + model
UPDATE cars
SET id = CONCAT(make, model);

-- Make id a primary key
ALTER TABLE cars
ADD CONSTRAINT id_pk PRIMARY KEY(id);

-- Have a look at the table
SELECT * FROM cars;
```

### Implementing relationships with foreign keys
- A foreign key (FK) points to the primary key (PK) of another table
- Domain of FK must be equal to domain of PK
- Each value of FK must exist in PK of the other table (FK constraint or "referential integrity")
- FKs are not actual keys because duplicates and null values allowed
foreign key prevents violations

### reference table with a foreign key
Table a should now refer to table b, via b_id, which points to id. a_fkey is, as usual, a constraint name you can choose on your own.

Pay attention to the naming convention employed here: Usually, a foreign key referencing another primary key with name id is named x_id, where x is the name of the referencing table in the singular form.

but becareful, inserting non-existing id violates foreign key constraint
```
ALTER TABLE a 
ADD CONSTRAINT a_fkey FOREIGN KEY (b_id) REFERENCES b (id);
```
example:
Add a foreign key on university_id column in professors that references the id column in universities.
Name this foreign key professors_fkey
```
-- Rename the university_shortname column
ALTER TABLE professors
RENAME COLUMN university_shortname TO university_id;

-- Add a foreign key on professors referencing universities
ALTER TABLE professors 
ADD CONSTRAINT professors_fkey FOREIGN KEY (university_id) REFERENCES universities (id);
```

### How to implement N:M-relationships
- Create a table
- Add foreign keys for every connected table
- Add additional attributes
```
CREATE TABLE affiliations (
professor_id integer REFERENCES professors (id),
organization_id varchar(256) REFERENCES organizations (id),
function varchar(256)
);
```
- No primary key!
- Possible PK = {professor_id, organization_id, function}

###  update columns of a table based on values in another table
```
UPDATE table_a
SET column_to_update = table_b.column_to_update_from
FROM table_b
WHERE condition1 AND condition2 AND ...;
This query does the following:
```
For each row in table_a, find the corresponding row in table_b where condition1, condition2, etc., are met.
Set the value of column_to_update to the value of column_to_update_from (from that corresponding row).
The conditions usually compare other columns of both tables, e.g. table_a.some_column = table_b.some_column. Of course, this query only makes sense if there is only one matching row in table_b.

example:
```
-- Update professor_id to professors.id where firstname, lastname correspond to rows in professors
UPDATE affiliations
SET professor_id = professors.id
FROM professors
WHERE affiliations.firstname = professors.firstname AND affiliations.lastname = professors.lastname;
```

### Referential integrity
- A record referencing another table must refer to an existing record in that table
- Specified between two tables
- Enforced through foreign keys

### Referential integrity violations
- Referential integrity from table A to table B is violated...
- ...if a record in table B that is referenced from a record in table A is deleted.
- ...if a record in table A referencing a non-existing record from table B is inserted.

Foreign keys prevent violations!

example: You defined a foreign key on professors.university_id that references universities.id, so referential integrity is said to hold from professors to universities.


### Dealing with violations
ON DELETE...
...NO ACTION: Throw an error
...CASCADE: Delete all referencing records
...RESTRICT: Throw an error
...SET NULL: Set the referencing column to NULL
...SET DEFAULT: Set the referencing column to its default value


option 1 (default):
```
CREATE TABLE a (
id integer PRIMARY KEY,
column_a varchar(64),
...,
b_id integer REFERENCES b (id) ON DELETE NO ACTION
);
```
option 2: 
auto delete record in b and a
```
CREATE TABLE a (
id integer PRIMARY KEY,
column_a varchar(64),
...,
b_id integer REFERENCES b (id) ON DELETE CASCADE
);
```

### Change the referential integrity behavior of a key
Altering a key constraint doesn't work with ALTER COLUMN. Instead, you have to delete the key constraint and then add a new one with a different ON DELETE behavior.

For deleting constraints, though, you need to know their name. This information is also stored in information_schema.
```
-- Identify the correct constraint name
SELECT constraint_name, table_name, constraint_type
FROM information_schema.table_constraints
WHERE constraint_type = 'FOREIGN KEY';

-- Drop the right foreign key constraint
ALTER TABLE affiliations
DROP CONSTRAINT affiliations_organization_id_fkey;

-- Add a new foreign key constraint from affiliations to organizations which cascades deletion
ALTER TABLE affiliations
ADD CONSTRAINT affiliations_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES organizations (id) ON DELETE CASCADE;
```

