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

### order by
The ORDER BY keyword is used to sort results in ascending or descending order according to the values of one or more columns.
By default ORDER BY will sort in ascending order. If you want to sort the results in descending order, you can use the DESC keyword
SELECT title
FROM films
ORDER BY release_year DESC;

ORDER BY can also be used to sort on multiple columns. It will sort by the first column specified, then sort by the next, then the next, and so on.
SELECT birthdate, name
FROM people
ORDER BY birthdate, name;

### groupby
GROUP BY always goes after the FROM clause!
SQL will return an error if you try to SELECT a field that is not in your GROUP BY clause without using it to calculate some kind of value about the entire group.
GROUP BY allows you to group a result by one or more columns, like so:
SELECT sex, count(*)
FROM employees
GROUP BY sex
ORDER BY count DESC;

### having clause
aggregate functions can't be used in WHERE clauses. 
that's where the HAVING clause comes in. For example,
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

### metadatabases / schema
holds information regarding your databases

in the entity-relationship diagram,
circle is attribute, square is entity.
separate entity (in each table) reduce redundancy with their respective attributes

### create tables
semicolon at the end impt before query!!

```
CREATE TABLE table_name (
 column_a data_type,
 column_b data_type,
 column_c data_type
);
```

datatype: text, numeric, char(5), etc mentioned later

### add columns to table
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

### find out what are the unique key
```
SELECT DISTINCT firstname,lastname
FROM professors
ORDER BY lastname;
```
gives say 500 records
if this
```
SELECT DISTINCT firstname,lastname,university_shortname
FROM professors
ORDER BY lastname;
```
also gives 500 records, university_shortname is not needed to uniquely identify the professor

can also use COUNT(DISTINCT())

### integrity constraints
Why constraints?
Constraints give the data structure
Constraints help with consistency, and thus data quality
Data quality is a business advantage / data science prerequisite
Enforcing is di cult, but PostgreSQL helps
1. Attribute constraints/ domain, e.g. data types on columns (Chapter 2)
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
3. char [ (x) ] : a fixed-length string of n characters
4. boolean : can only take three states, e.g. TRUE , FALSE and NULL (unknown)
5. date , time and timestamp : various formats for date and time calculations. stored as YYYY-MM-DD
6. numeric : arbitrary precision numbers, e.g. 3.1457. 
eg numeric(3,2) is with precision of 3 and scale of 2, meaning numbers with a total of three and two digits are allowed.
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


NULL!=NULL 

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

### remove not-null after table created
```
ALTER TABLE students
ALTER COLUMN home_phone
DROP NOT NULL;
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
example 3: this is still one primary key. just that its a combination of two columns
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
created artificial primary key (instead of using say combination of two columns like example 3 above)
- Primary keys should be built from as few columns as possible
- Primary keys should never change over time

Adding a surrogate key with serial data type.
add an incremental number that is unique to become a primary key
```
ALTER TABLE cars
ADD COLUMN id serial PRIMARY KEY;
INSERT INTO cars
VALUES ('Volkswagen', 'Blitz', 'black');
```

### create primary key with 2 columns via concat instead of using say serial surrogate
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

### 1:N relationship, reference table with a foreign key
Table a should now refer to table b, via a_id, which points to b_id. a_fkey is, as usual, a constraint name you can choose on your own.

Pay attention to the naming convention employed here: Usually, a foreign key referencing another primary key with name id is named x_id, where x is the name of the referencing table in the singular form.

but becareful, inserting non-existing id violates foreign key constraint
```
ALTER TABLE a 
ADD CONSTRAINT a_fkey FOREIGN KEY (a_id) REFERENCES b (b_id);
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

While foreign keys and primary keys are not strictly necessary for join queries, they greatly help by telling you what to expect. For instance, you can be sure that records referenced from table A will always be present in table B – so a join from table A will always find something in table B. If not, the foreign key constraint would be violated.


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

before the migration of the data to remove away the affiliations table, transform the table in place first before creating a function table to link the relationship

###  update columns of a table based on values in another table
this is for the linking function table
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
ON DELETE:
. NO ACTION: Throw an error
. CASCADE: Delete all referencing records
. RESTRICT: Throw an error
. SET NULL: Set the referencing column to NULL
. SET DEFAULT: Set the referencing column to its default value.only works if default value specified 


option 1 (default delete no action is automatically append to foreign key):
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

### Change foreign key referential violation constraint
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

-- Delete an organization 
DELETE FROM organizations 
WHERE id = 'CUREM';

-- Check that no more affiliations with this organization exist
SELECT * FROM affiliations
WHERE organization_id = 'CUREM';
```

### joining all tables
can notice for join university, is not on affiliations but can be on other tables such as professors 
```
-- Join all tables
SELECT *
FROM affiliations
JOIN professors
ON affiliations.professor_id = professors.id
JOIN organizations
ON affiliations.organization_id = organizations.id
JOIN universities
ON professors.university_id = universities.id;
```
## Intermediate SQL

### case statements
great for
- Categorizing data
- Filtering data
- Aggregating data

```
CASE WHEN x = 1 THEN 'a'
WHEN x = 2 THEN 'b'
ELSE 'c' END AS new_column
```

example:
```
SELECT
id,
home_goal,
away_goal,
CASE WHEN home_goal > away_goal THEN 'Home Team Win'
WHEN home_goal < away_goal THEN 'Away Team Win'
ELSE 'Tie' END AS outcome
FROM match
WHERE season = '2013/2014';
| id | home_goal | away_goal |
```

```
SELECT date, hometeam_id, awayteam_id,
  CASE WHEN hometeam_id = 8455 AND home_goal > away_goal
    THEN 'Chelsea home win!'
  WHEN awayteam_id = 8455 AND home_goal < away_goal
    THEN 'Chelsea away win!'
  ELSE 'Loss or tie :(' END AS outcome
FROM match
WHERE hometeam_id = 8455 OR awayteam_id = 8455;
```

### WHERE IN
```
SELECT
	-- Select the team long name and team API id
	team_long_name,
	team_api_id
FROM teams_germany
-- Only include FC Schalke 04 and FC Bayern Munich
WHERE team_long_name IN ('FC Schalke 04','FC Bayern Munich');
```

### NULL in CASE statements
``` 
SELECT date,
CASE WHEN date > '2015-01-01' THEN 'More Recently'
WHEN date < '2012-01-01' THEN 'Older'
END AS date_category
FROM match;
```

same as 
```
SELECT date,
CASE WHEN date > '2015-01-01' THEN 'More Recently'
WHEN date < '2012-01-01' THEN 'Older'
ELSE NULL END AS date_category
FROM match;
```

### Filter CASE in WHERE
- to exclude data you dont want.
- you can use the CASE statement as a filtering column like any other column in your database. The only difference is that you don't alias the statement in WHERE.
- copy everything in CASE and put inside WHERE.
```
-- Select the season, date, home_goal, and away_goal columns
SELECT  
	season,
    date,
	home_goal,
	away_goal
FROM matches_italy
WHERE 
-- Exclude games not won by Bologna
	CASE WHEN hometeam_id = 9857 AND home_goal > away_goal THEN 'Bologna Win'
		WHEN awayteam_id = 9857 AND away_goal > home_goal THEN 'Bologna Win' 
		END IS NOT NULL;
```

### CASE WHEN with COUNT
end with column instead of string of text

```
SELECT
season,
COUNT(CASE WHEN hometeam_id = 8650
AND home_goal > away_goal
THEN id END) AS home_wins
FROM match
GROUP BY season;
```

```
SELECT 
	c.name AS country,
    -- Count games from the 2012/2013 season
	COUNT(CASE WHEN m.season = '2012/2013' 
        	THEN m.id ELSE NULL END) AS matches_2012_2013
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY country;
```

### CASE WHEN with SUM
```
SELECT 
	c.name AS country,
    -- Sum the total records in each season where the home team won
	SUM(CASE WHEN m.season = '2012/2013' AND m.home_goal > m.away_goal 
        THEN 1 ELSE 0 END) AS matches_2012_2013,
 	SUM(CASE WHEN m.season = '2013/2014' AND m.home_goal > m.away_goal  
        THEN 1 ELSE 0 END) AS matches_2013_2014,
	SUM(CASE WHEN m.season = '2014/2015' AND m.home_goal > m.away_goal 
        THEN 1 ELSE 0 END) AS matches_2014_2015
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY country;
```


### Percentages with CASE and AVG
```
SELECT 
	c.name AS country,
    -- Round the percentage of tied games to 2 decimal points
	ROUND(AVG(CASE WHEN m.season='2013/2014' AND m.home_goal = m.away_goal THEN 1
			 WHEN m.season='2013/2014' AND m.home_goal != m.away_goal THEN 0
			 END),2) AS pct_ties_2 013_2014,
	ROUND(AVG(CASE WHEN m.season='2014/2015' AND m.home_goal = m.away_goal THEN 1
			 WHEN m.season='2014/2015' AND m.home_goal != m.away_goal THEN 0
			 END),2) AS pct_ties_2014_2015
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
GROUP BY country;
```

### What is a subquery?
A query nested inside another query
```
SELECT column
FROM (SELECT column
FROM table) AS subquery;
```

### Why subqueries?
- Comparing groups to summarized values
How did Liverpool compare to the English Premier League's average performance for that year?
- Reshaping data
What is the highest monthly average of goals scored in the Bundesliga?
- Combining data that cannot be joined
How do you get both the home and away team names into a table of match results?

### Subqueries in the WHERE clause
You can filter data based on single, scalar values using a subquery in ways you cannot by using WHERE statements or joins.

In addition to filtering using a single-value (scalar) subquery, you can create a list of values in a subquery to filter data based on a complex set of conditions. This type of subquery generates a one column reference list for the main query. As long as the values in your list match a column in your main query's table, you don't need to use a join -- even if the list is from a separate table.
```
SELECT AVG(home_goal) FROM match;
1.56091291478423
SELECT date, hometeam_id, awayteam_id, home_goal, away_goal
FROM match
WHERE season = '2012/2013'
AND home_goal > 1.56091291478423;
```

same as
```
SELECT date, hometeam_id, awayteam_id, home_goal, away_goal
FROM match
WHERE season = '2012/2013'
AND home_goal > (SELECT AVG(home_goal)
FROM match);
```

NOT IN
```
SELECT 
	-- Select the team long and short names
	team_long_name,
	team_short_name
FROM team 
-- Exclude all values from the subquery
WHERE team_api_id NOT IN
     (SELECT DISTINCT hometeam_id  FROM match);
```

### Subqueries in FROM
- Restructure and transform your data
  - Transforming data from long to wide before selecting
  - Prefiltering data
- Calculating aggregates of aggregates
  - Which 3 teams has the highest average of home goals scored?
  1. Calculate the AVG for each team
  2. Get the 3 highest of the AVG values

example:
```
SELECT team, home_avg
FROM (SELECT
        t.team_long_name AS team,
        AVG(m.home_goal) AS home_avg
      FROM match AS m
      LEFT JOIN team AS t
      ON m.hometeam_id = t.team_api_id
      WHERE season = '2011/2012'
      GROUP BY team) AS subquery
ORDER BY home_avg DESC
LIMIT 3;
```
example:
```
SELECT
	-- Select country name and the count match IDs
    c.name AS country_name,
    COUNT(*) AS matches
FROM country AS c
-- Inner join the subquery onto country
-- Select the country id and match id columns
INNER JOIN (SELECT country_id, id 
           FROM match
           -- Filter the subquery by matches with 10+ goals
           WHERE (home_goal + away_goal) >=10) AS sub
ON c.id = sub.country_id
GROUP BY country_name;
```
example:
```
SELECT
	-- Select country, date, home, and away goals from the subquery
    country,
    date,
    home_goal,
    away_goal
FROM 
	-- Select country name, date, and total goals in the subquery
	(SELECT c.name AS country, 
     	    m.date, 
     		m.home_goal, 
     		m.away_goal,
           (m.home_goal + m.away_goal) AS total_goals
    FROM match AS m
    LEFT JOIN country AS c
    ON m.country_id = c.id) AS subq
-- Filter by total goals scored in the main query
WHERE total_goals >= 10;
```

### Subqueries in SELECT
- Returns a single value
  - Include aggregate values to compare to individual values
- Used in mathematical calculations
  - Deviation from the average

```
SELECT
  eason,
  COUNT(id) AS matches,
  12837 as total_matches
FROM match
GROUP BY season;
```

or use subquery
```
SELECT
  season,
  COUNT(id) AS matches,
  (SELECT COUNT(id) FROM match) as total_matches
FROM match
GROUP BY season;
```

```
SELECT
  date,
  (home_goal + away_goal) AS goals,
  (home_goal + away_goal) -
    (SELECT AVG(home_goal + away_goal)
    FROM match
    WHERE season = '2011/2012') AS diff
FROM match
WHERE season = '2011/2012';
```

REMEMBER TO PROPERLY FILTER BOTH MAIN AND SUBQUERY
```
SELECT
  date,
  (home_goal + away_goal) AS goals,
  (home_goal + away_goal) -
    (SELECT AVG(home_goal + away_goal)
    FROM match
    WHERE season = '2011/2012') AS diff
FROM match
WHERE season = '2011/2012';
```

example:
```
SELECT
	-- Select the league name and average goals scored
	name AS league,
	ROUND(AVG(m.home_goal + m.away_goal),2) AS avg_goals,
    -- Subtract the overall average from the league average
	ROUND(AVG(m.home_goal + m.away_goal) - 
		(SELECT AVG(home_goal + away_goal)
		 FROM match 
         WHERE season = '2013/2014'),2) AS diff
FROM league AS l
LEFT JOIN match AS m
ON l.country_id = m.country_id
-- Only include 2013/2014 results
WHERE season = '2013/2014'
GROUP BY l.name;
```

### comment in sql
/* */

### style for sql formatting
holywell's sql style guide

### example of subquery in SELECt, WHERE, FROM
```
SELECT 
	-- Select the stage and average goals from s
	s.stage,
    ROUND(s.avg_goals,2) AS avg_goal,
    -- Select the overall average for 2012/2013
    (SELECT AVG(home_goal + away_goal) FROM match WHERE season = '2012/2013') AS overall_avg
FROM 
	-- Select the stage and average goals in 2012/2013 from match
	(SELECT
		 stage,
         AVG(home_goal + away_goal) AS avg_goals
	 FROM match
	 WHERE season = '2012/2013'
	 GROUP BY stage) AS s
WHERE 
	-- Filter the main query using the subquery
	s.avg_goals > (SELECT AVG(home_goal + away_goal) 
                    FROM match WHERE season = '2012/2013');
```

### Correlated subquery
- Uses values from the outer query to generate a result
- Re-run for every row generated in the final data set
- Used for advanced joining, filtering, and evaluating data

### Simple vs. correlated subqueries
1. Simple Subquery
  - Can be run independently from the main query
  - Evaluated once in the whole
query
2. Correlated subqueries 
  - are subqueries that reference one or more columns in the main query. Correlated subqueries depend on information in the main query to run, and thus, cannot be executed on their own.
  - Evaluated in loops
  - Correlated subqueries are evaluated in SQL once per row of data retrieved -- a process that takes a lot more computing power and time than a simple subquery.

example of using join:
What is the average number
of goals scored in each
country?
```
SELECT
  c.name AS country,
  AVG(m.home_goal + m.away_goal)
    AS avg_goals
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
GROUP BY country;
```

example of using correlated subqueries for the same thing:
```
SELECT
  c.name AS country,
  (SELECT
    AVG(home_goal + away_goal)
  FROM match AS m
  WHERE m.country_id = c.id)
    AS avg_goals
FROM country AS c
GROUP BY country;
```


```
SELECT 
	-- Select country ID, date, home, and away goals from match
	main.country_id,
    main.date,
    main.home_goal, 
    main.away_goal
FROM match AS main
WHERE 
	-- Filter the main query by the subquery
	(home_goal + away_goal) > 
        (SELECT AVG((sub.home_goal + sub.away_goal) * 3)
         FROM match AS sub
         -- Join the main query to the subquery in WHERE
         WHERE main.country_id = sub.country_id);
```

```
SELECT 
	-- Select country ID, date, home, and away goals from match
	main.country_id,
    main.date,
    main.home_goal,
    main.away_goal
FROM match AS main
WHERE 
	-- Filter for matches with the highest number of goals scored
	(home_goal + away_goal) =
        (SELECT MAX(sub.home_goal + sub.away_goal)
         FROM match AS sub
         WHERE main.country_id = sub.country_id
               AND main.season = sub.season);
```

### Nested subquery
can be correlated or uncorrelated or both for the inter and outer subquery
```
SELECT
  EXTRACT(MONTH FROM date) AS month,
  SUM(m.home_goal + m.away_goal) AS total_goals,
  SUM(m.home_goal + m.away_goal) -
  (SELECT AVG(goals)
  FROM (SELECT
      EXTRACT(MONTH FROM date) AS month,
      SUM(home_goal + away_goal) AS goals
    FROM match
    GROUP BY month) AS s) AS diff
FROM match AS m
GROUP BY month;

| month | goals | diff |
|-------|-------|----------|
| 01 | 5821 | -36.25 |
| 02 | 7448 | 1590.75 |
| 03 | 7298 | 1440.75 |
| 04 | 8145 | 2287.75 |
```

### Correlated nested subquery
a nested subquery's components can be executed independently of the outer query, while a correlated subquery requires both the outer and inner subquery to run and produce results.

example:
What is the each country's average goals scored in the 2011/2012 season?
got second nested subquery inside SELECT
```
SELECT
  c.name AS country,
  (SELECT AVG(home_goal + away_goal)
  FROM match AS m
  WHERE m.country_id = c.id -- Correlates with main query
      AND id IN (
        SELECT id -- Begin inner subquery
        FROM match
        WHERE season = '2011/2012')) AS avg_goals
FROM country AS c
GROUP BY country;
```


### nested simple subqueries 
example:
nested subquery to examine the highest total number of goals in each season, overall, and during July across all seasons.
```
SELECT
	-- Select the season and max goals scored in a match
	season,
    MAX(home_goal + away_goal) AS max_goals,
    -- Select the overall max goals scored in a match
    -- this is a value
   (SELECT MAX(home_goal + away_goal) FROM match) AS overall_max_goals,
   -- Select the max number of goals scored in any match in July
   -- this is a value
   (SELECT MAX(home_goal + away_goal) 
    FROM match
    WHERE id IN (
          SELECT id FROM match WHERE EXTRACT(MONTH FROM date) = 07)) AS july_max_goals
FROM match
GROUP BY season;
```

### Nest a subquery in FROM
example:
What's the average number of matches per season where a team scored 5 or more goals? How does this differ by country?
```
SELECT
	c.name AS country,
    -- Calculate the average matches per season
	AVG(outer_s.matches) AS avg_seasonal_high_scores
FROM country AS c
-- Left join outer_s to country
LEFT JOIN (
  SELECT country_id, season,
         COUNT(id) AS matches
  FROM (
    SELECT country_id, season, id
	FROM match
	WHERE home_goal >= 5 OR away_goal >= 5) AS inner_s
  -- Close parentheses and alias the subquery
  GROUP BY country_id, season) AS outer_s
ON c.id = outer_s.country_id
GROUP BY country;
```

### Common Table Expressions (CTEs)
- Table declared before the main query
- Named and referenced later in FROM statement

```
WITH cte AS (
SELECT col1, col2
FROM table)
SELECT
AVG(col1) AS avg_col
FROM cte;
```

more than 1 ctes:
```
WITH s1 AS (
  SELECT country_id, id
  FROM match
  WHERE (home_goal + away_goal) >= 10),
s2 AS ( -- New subquery
  SELECT country_id, id
  FROM match
  WHERE (home_goal + away_goal) <= 1
)
SELECT
  c.name AS country,
  COUNT(s1.id) AS high_scores,
  COUNT(s2.id) AS low_scores -- New column
FROM country AS c
INNER JOIN s1
ON c.id = s1.country_id
INNER JOIN s2 -- New join
ON c.id = s2.country_id
GROUP BY country;
```

```
-- Set up your CTE
WITH match_list AS (
  -- Select the league, date, home, and away goals
    SELECT 
  		l.name AS league, 
     	date, 
  		m.home_goal, 
  		m.away_goal,
       (m.home_goal + m.away_goal) AS total_goals
    FROM match AS m
    LEFT JOIN league as l ON m.country_id = l.id)
-- Select the league, date, home, and away goals from the CTE
SELECT league, date, home_goal, away_goal
FROM match_list
-- Filter by total goals
WHERE total_goals >=10;
```

```
-- Set up your CTE
WITH match_list AS (
    SELECT 
  		country_id,
  	   (home_goal + away_goal) AS goals
    FROM match
  	-- Create a list of match IDs to filter data in the CTE
    WHERE id IN (
       SELECT id
       FROM match
       WHERE season = '2013/2014' AND EXTRACT(MONTH FROM date) = 08))
-- Select the league name and average of goals in the CTE
SELECT 
	name,
    AVG(goals)
FROM league AS l
-- Join the CTE onto the league table
LEFT JOIN match_list ON l.id = match_list.country_id
GROUP BY l.name;
```

### Why use CTEs?
- Executed once
- CTE is then stored in memory
- Improves query performance
- Improving organization of queries
- Referencing other CTEs
- Referencing itself ( SELF JOIN )

### comparing joins, subqueries, ctes
you can use multiple techniques in SQL to answer the same question.

Joins:
- Combine 2+ tables
    - Simple
    - operations/aggregations

Correlated Subqueries:
- Match subqueries & tables
    - Avoid limits of joins 
      example cant join 2 separate column to one single column
- High processing time

Multiple/Nested Subqueries:
- Multi-step transformations
- Improve accuracy and reproducibility

Common Table Expressions:
- Organize subqueries sequentially
- Can reference other CTEs
- CTEs are stored in memory, reducing query run time

### subquery, join and cte example
```
SELECT
	m.date,
    -- Get the home and away team names
    hometeam,
    awayteam,
    m.home_goal,
    m.away_goal
FROM match AS m

-- Join the home subquery to the match table
LEFT JOIN(
  SELECT match.id, team.team_long_name AS hometeam
  FROM match
  LEFT JOIN team
  ON match.hometeam_id = team.team_api_id) AS home
ON home.id = m.id

-- Join the away subquery to the match table
LEFT JOIN (
  SELECT match.id, team.team_long_name AS awayteam
  FROM match
  LEFT JOIN team
  -- Get the away team ID in the subquery
  ON match.awayteam_id = team.team_api_id) AS away
ON  away.id= m.id;
```

same as the following using just subquery
```
SELECT
    m.date,
    (SELECT team_long_name
     FROM team AS t
     WHERE t.team_api_id = m.hometeam_id) AS hometeam,
    -- Connect the team to the match table
    (SELECT team_long_name
     FROM team AS t
     WHERE t.team_api_id = m.awayteam_id) AS awayteam,
    -- Select home and away goals
     home_goal,
     away_goal
FROM match AS m;
```

same as the following using ctes:
```
WITH home AS (
  SELECT m.id, m.date, 
  		 t.team_long_name AS hometeam, m.home_goal
  FROM match AS m
  LEFT JOIN team AS t 
  ON m.hometeam_id = t.team_api_id),
-- Declare and set up the away CTE
away AS (
  SELECT m.id, m.date, 
  		 t.team_long_name AS awayteam, m.away_goal
  FROM match AS m
  LEFT JOIN team AS t 
  ON m.awayteam_id = t.team_api_id)
-- Select date, home_goal, and away_goal
SELECT 
	home.date,
    home.hometeam,
    away.awayteam,
    home.home_goal,
    away.away_goal
-- Join away and home on the id column
FROM home
INNER JOIN away
ON home.id = away.id;
```


### cte with join example
```
-- Declare the home CTE
WITH home as (
	SELECT m.id, t.team_long_name AS hometeam
	FROM match AS m
	LEFT JOIN team AS t 
	ON m.hometeam_id = t.team_api_id)
-- Select everything from home
SELECT *
FROM home;
```

same as:
```
SELECT 
	-- Select match id and team long name
    m.id, 
    t.team_long_name AS hometeam
FROM match AS m
-- Join team to match using team_api_id and hometeam_id
LEFT JOIN team AS t 
ON t.team_api_id = m.hometeam_id;
```

### window function
Working with aggregate values, requires you to use GROUP BY with all non-aggregate
columns.

but window function can work around it. window functions allow getting aggregates without having to group data.

- Perform calculations on an already generated result set (a window)
- Aggregate calculations
  - Similar to subqueries in SELECT
  - Running totals, rankings, moving averages


- Processed after every part of query except ORDER BY
  - Uses information in result set rather than database
- Available in PostgreSQL, Oracle, MySQL, SQL Server...
  - ...but NOT SQLite


### OVER() clause in window function
allows you to pass an aggregate function down a data set, similar to subqueries in SELECT. The OVER() clause offers significant benefits over subqueries in select -- namely, your queries will run faster, and the OVER() clause has a wide range of additional functions and clauses you can include with

```
SELECT
  date,
  (home_goal + away_goal) AS goals,
  AVG(home_goal + away_goal) OVER() AS overall_avg
FROM match
WHERE season = '2011/2012';
```

without OVER() looks like this:
```
SELECT
date,
(home_goal + away_goal) AS goals,
(SELECT AVG(home_goal + away_goal)
  FROM match
  WHERE season = '2011/2012') AS overall_avg
FROM match
WHERE season = '2011/2012';
```

### RANK() clause in window function
Window functions allow you to create a RANK of information according to any variable you want to use to sort your data. When setting this up, you will need to specify what column/calculation you want to use to calculate your rank. This is done by including an ORDER BY clause inside the OVER() clause. 

```
SELECT
  date,
  (home_goal + away_goal) AS goals,
  RANK() OVER(ORDER BY home_goal + away_goal DESC) AS goals_rank
FROM match
WHERE season = '2011/2012';


| date | goals | goals_rank |
|------------|-------|------------|
| 2012-04-28 | 0 | 1 |
| 2011-12-26 | 0 | 1 |
| 2011-09-10 | 0 | 1 |

```

```
SELECT 
	-- Select the league name and average goals scored
	name AS league,
    AVG(m.home_goal + m.away_goal) AS avg_goals,
    -- Rank leagues in descending order by average goals
    RANK() OVER(ORDER BY AVG(m.home_goal + m.away_goal) DESC) AS league_rank
FROM league AS l
LEFT JOIN match AS m 
ON l.id = m.country_id
WHERE m.season = '2011/2012'
GROUP BY l.name
-- Order the query by the rank you created
ORDER BY league_rank;
```

### OVER and PARTITION BY
- Calculate separate values for diferent categories
- Calculate different calculations in the same column

```
AVG(home_goal) OVER(PARTITION BY season)
```

PARTITION BY considerations
- Can partition data by 1 or more columns
- Can partition aggregate calculations, ranks, etc


The PARTITION BY clause allows you to calculate separate "windows" based on columns you want to divide your results. For example, you can create a single column that calculates an overall average of goals scored for each season.

example partition by one column:
```
SELECT
  date,
  (home_goal + away_goal) AS goals,
  AVG(home_goal + away_goal) OVER(PARTITION BY season) AS season_avg
FROM match;
```

example partition by multiple columns:
```
SELECT
  c.name,
  m.season,
  (home_goal + away_goal) AS goals,
  AVG(home_goal + away_goal)
    OVER(PARTITION BY m.season, c.name) AS season_ctry_avg
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id

| name | season | goals | season_ctry_avg |
|-------------|-----------|-----------|-----------------|
| Belgium | 2011/2012 | 1 | 2.88 |
| Netherlands | 2014/2015 | 1 | 3.08 |
| Belgium | 2011/2012 | 1 | 2.88 |

```

```
SELECT
	date,
	season,
	home_goal,
	away_goal,
	CASE WHEN hometeam_id = 8673 THEN 'home' 
		 ELSE 'away' END AS warsaw_location,
    -- Calculate the average goals scored partitioned by season
    AVG(home_goal) OVER(PARTITION BY season) AS season_homeavg,
	AVG(away_goal) OVER(PARTITION BY season) AS season_awayavg
FROM match
-- Filter the data set for Legia Warszawa matches only
WHERE 
	hometeam_id = 8673
    OR awayteam_id = 8673
ORDER BY (home_goal + away_goal) DESC;
```

```
SELECT 
	date,
	season,
	home_goal,
	away_goal,
	CASE WHEN hometeam_id = 8673 THEN 'home' 
         ELSE 'away' END AS warsaw_location,
	-- Calculate average goals partitioned by season and month
    AVG(home_goal) OVER(PARTITION BY season, 
         	EXTRACT(MONTH FROM date)) AS season_mo_home,
    AVG(away_goal) OVER(PARTITION BY season, 
            EXTRACT(MONTH FROM date)) AS season_mo_away
FROM match
WHERE 
	hometeam_id = 8673 
    OR awayteam_id = 8673
ORDER BY (home_goal + away_goal) DESC;
```

### Sliding Windows
- Perform calculations relative to the current row
- Can be used to calculate running totals, sums, averages, etc
- Can be partitioned by one or more columns

Sliding Window Keywords:
Sliding windows allow you to create running calculations between any two points in a window using functions such as PRECEDING, FOLLOWING, and CURRENT ROW.

'''
ROWS BETWEEN <start> AND <finish>
'''
PRECEDING: specify number of rows before current row
FOLLOWING: specify number of rows after current row
UNBOUNDED PRECEDING: every row since the beginning
UNBOUNDED FOLLOWING: every row since the end
CURRENT ROW: stop calculation at current row

```
SELECT
  date,
  home_goal,
  away_goal,
  SUM(home_goal)
    OVER(ORDER BY date ROWS BETWEEN
      UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM match
WHERE hometeam_id = 8456 AND season = '2011/2012';

```
| date | home_goal | away_goal | running_total |
|------------|-----------|-----------|---------------|
| 2011-08-15 | 4 | 0 | 4 |
| 2011-09-10 | 3 | 0 | 7 |
| 2011-09-24 | 2 | 0 | 9 |
| 2011-10-15 | 4 | 1 | 13 |

```
SELECT date,
  home_goal,
  away_goal,
  SUM(home_goal)
    OVER(ORDER BY date
    ROWS BETWEEN 1 PRECEDING
    AND CURRENT ROW) AS last2
FROM match
WHERE hometeam_id = 8456
  AND season = '2011/2012';
```

In this exercise, you will slightly modify the query from the previous exercise by sorting the data set in reverse order and calculating a backward running total from the CURRENT ROW to the end of the data set (earliest record).
```
SELECT 
	-- Select the date, home goal, and away goals
	date,
    home_goal,
    away_goal,
    -- Create a running total and running average of home goals
    SUM(home_goal) OVER(ORDER BY date DESC
         ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS running_total,
    AVG(home_goal) OVER(ORDER BY date DESC
         ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS running_avg
FROM match
WHERE 
	awayteam_id = 9908 
    AND season = '2011/2012';
```

### combining case,subqueries,cte,window functions
generate a list of matches in which Manchester United was defeated during the 2014/2015 English Premier League season

```
-- Set up the home team CTE
WITH home AS (
  SELECT m.id, t.team_long_name,
	  CASE WHEN m.home_goal > m.away_goal THEN 'MU Win'
		   WHEN m.home_goal < m.away_goal THEN 'MU Loss' 
  		   ELSE 'Tie' END AS outcome
  FROM match AS m
  LEFT JOIN team AS t ON m.hometeam_id = t.team_api_id),
-- Set up the away team CTE
away AS (
  SELECT m.id, t.team_long_name,
	  CASE WHEN m.home_goal > m.away_goal THEN 'MU Win'
		   WHEN m.home_goal < m.away_goal THEN 'MU Loss' 
  		   ELSE 'Tie' END AS outcome
  FROM match AS m
  LEFT JOIN team AS t ON m.awayteam_id = t.team_api_id)
-- Select team names, the date and goals
SELECT DISTINCT
    date,
    home.team_long_name AS home_team,
    away.team_long_name AS away_team,
    m.home_goal,
    m.away_goal
-- Join the CTEs onto the match table
FROM match AS m
LEFT JOIN home ON m.id = home.id
LEFT JOIN away ON m.id = away.id
WHERE m.season = '2014/2015'
      AND (home.team_long_name = 'Manchester United' 
           OR away.team_long_name = 'Manchester United');
```
