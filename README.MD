
# Guild Education Data Engineering Project Overview  
  
### Project Goals  
- Design a data model that can be used to answer a series of questions.  
- Implement a program that transforms the input data into a form usable by the data model  
- Explain how you would scale this pipeline  
  
The designed data model must be able to at least answer the following questions:  
- Production Company Details:  
    - budget per year  
    - revenue per year  
    - profit per year  
    - releases by genre per year  
    - average popularity of produced movies per year  
- Movie Genre Details:  
    - most popular genre by year  
    - budget by genre by year  
    - revenue by genre by year  
    - profit by genre by year  
  
### File Deliverables  
Here is a list of the files/directories created for this project and a description of each:  
- Directory - `movie_data_unscaled`  
  - `__init__.py` - script initialization file  
  - `main.py` - Python file to execute the `movie_processor` function from the `MovieDataProcessor` class
  - `params.py` - file containing default parameters for the base environment (e.g. Region, AWS Access Key)
  - `requirements.txt` - file containing required Python packages for the code  
  - `Dockerfile` - Dockerfile for use when creating Lambda Layer for Python packages
  - `movieDataProcessing.py` - script that takes in the five main files from the `the-movies-dataset.zip` file   
    (`movies_metadata.csv`, `keywords.csv`, `links.csv`, `credits.csv`, & `ratings.csv`), cleanses the data, creates a  
    new "aggregated" dataframe to perform desired SQL functions on, then uploads cleansed/new dataframes to S3 as .csv  
    files along with the error log file  
        
- Directory - `movie_data_scaled`  
  - `__init__.py` - script initialization file  
  - `main.py` - Python file to execute the `movie_processor` function from the `MovieDataProcessorScaled` class  
  - `params.py` - file containing default parameters for the base environment (e.g. Region, AWS Access Key)
  - `requirements.txt` - file containing required Python packages for the code  
  - `Dockerfile` - Dockerfile for use when creating Lambda Layer for Python packages
  - `movieDataProcessingScaled.py` - script that takes in the five main files from the `the-movies-dataset.zip` file   
    (`movies_metadata.csv`, `keywords.csv`, `links.csv`, `credits.csv`, & `ratings.csv`), cleanses the data, creates a  
    new "aggregated" dataframe to perform desired SQL functions on, then uploads cleansed/new dataframes to S3 as   
    .json files along with the error log file  
    
### Building/Running the Code  
AWS Lambda will be the deployment method for this code. 
- AWS Lambda Deployment Instructions
  - Create a new Lambda function to run with Python 3.8   
  - Upload the necessary files to the Lambda directory (`__init__.py`, `movieDataProcessingScaled.py`, `main.py`)  
 - In the `lambda_function.py` file, put the code:  
```python
import os  
  
def lambda_handler(event, context):
    print(f"Incoming Event: {event}")  
    os.system('python3 main.py') 
```

 - Create a new S3 trigger by clicking the "Add trigger" button with the following configuration:  
	 - Bucket - `com.guild.us-west-2.public-data` 
	 - Prefix - `project-data/` 
	 - Suffix - `the-movies-dataset.zip`  
- (Optional) Create a new destination for Lambda logs to help with error handling by clicking the "Add destination" button
- Environment variables (e.g. region) can be set under the "Configuration" tab for the function
	
When using AWS Lambda, we need to go through a process to install required Python packages...  
- Create a Dockerfile with a configuration like:  
```shell  
FROM amazonlinux:2.0.20191016.0RUN yum install -y python38 && \  
yum install -y python3-pip && \  
yum install -y zip && \  
yum clean allRUN python3.8 -m pip install --upgrade pip && \  
python3.8 -m pip install virtualenv  
```
- Create Dockerfile with a tag
```shell
usr> docker build -f "<filename>.Dockerfile" -t lambdalayer:latest .
```
- Run/bash into container
```shell
usr> docker run -it --name lambdalayer lambdalayer:latest bash
```
- Create a new virtual environment
```shell
bash> python3.8 -m venv movies
```
- Install required packages  (example of moving current directory and using `requirements.txt`)
```shell  
bash> source pandas/bin/activate
(movies) bash> cd ./GitHub/data-engineering-project/movie_data_scaled  
(movies) bash> pip install -r requirements.txt  
(movies) bash> deactivate
```  
- Zip the folder, exit container, and copy to local environment
```shell
bash> zip -r python.zip ./python/
usr> docker cp lambdalayer:python.zip ./Desktop/
```
- Create a new Lambda Layer in AWS, check the `Upload a .zip file` option, and set the runtime to Python 3.8
        
### Scaling  
In order to handle scaling, I created a separate `movieDataProcessingScaled.py` version of the base   
`movieDataProcessing.py`. The unscaled `MovieDataProcessor` simply takes in the entire .csv files as their own full  dataframes and outputs full .csv files. `MovieDataProcessorScaled` instead performs scaling at two levels. The first  level is through the actual pandas file intake, which will be performed in chunks of 10,000 for each of the five data  files being ingested, with each chunk being iterated over and put through the cleansing and uploading process. The  second level is through multi-part, multi-threaded JSON file uploads using Boto3, with the current threshold set at 20mb, which can be further adjusted for scaling.  

Another way to help with scaling, since the code would be run through an AWS Lambda function, is to use the Application Auto Scaling and Lambda API's and define certain conditions. It allows for you to configure two types of concurrency:
- Reserved concurrency – creates a pool of requests that can only be used by its function, and also prevents its function from using unreserved concurrency.
- Provisioned concurrency – initializes a requested number of execution environments so that they are prepared to respond to your function's invocations.

Setting the configuration of concurrency for AWS Lambda allows for automatic scaling upon request call. There are also a few different scaling methods:
- Target tracking scaling - Scale a resource based on a target value for a specific CloudWatch metric.
- Step scaling - Scale a resource based on a set of scaling adjustments that vary based on the size of the alarm breach.
- Scheduled scaling - Scale a resource based on the date and time.

AWS Lambda allows for function log output through the AWS Lambda UI (briefly discussed in the `Building/Running the Code` section of this README) which can help with monitoring and error handling.

Scaling with AWS Lambda can lead to some issues, however. These include:
- Potential for the code base to become hard to manage/unmanageable
	- Each Lambda function needs a local copy of all the code that will be run, the code library sharing can become increasingly complex
-  AWS Lambda has built-in concurrency limits, which can lead to code not responding as the usage increases
-  Extra attention may need to be paid to file operations due to the nature of Lambda deployment
-  Gaps can appear between the domain of the business issue being addressed by the code and the actual code itself (technical/code drift)
	- As new information is received, new requirements are created
  
### Data Model
![Database Diagram](https://github.com/francescabeller/data-engineer-project/blob/master/guild-project-diagram.png)
Table Descriptions:
- `movies_master` - master table holding cleaned data from `movies_metadata.csv`
- `movies_aggregated`
	- new table generated from the data in `movies_master`
	- splits the JSON data in the `genre` and `production_companies` columns and creates a new row for each combination of `genre`/`production_company` in order to perform aggregation functions
- `keywords_master` - splits the JSON data from `keywords.csv` and creates a new row for each keyword connected to a `movie_id`
- `credits_master` - master table for basic clean data from `credits.csv`
- `cast_master` - new table generated from data in `credits_master` which takes the `cast` JSON column connected to a `movie_id` and splits each individual dictionary into a separate row and separate columns for each dictionary key:value
-  `crew_master` - new table generated from data in `credits_master` which takes the `crew` JSON column connected to a `movie_id` and splits each individual dictionary into a separate row and separate columns for each dictionary key:value
- `ratings_master` - master table for basic clean data from `ratings.csv`
- `links_master` - master table for basic clean data from `links.csv`


Here is an example row of data for each of the tables we will be using...

`MOVIES_MASTER`
id | adult | belongs_to_collection | budget | genres | homepage | imdb_id | original_language | original_title | overview | popularity | poster_path | production_companies | production_countries | release_date | revenue | runtime | spoken_language | status | tagline | title | video | vote_average | vote_count 
-- | ----- | ---------------------------- | --------|---------|--------------|----------|-----------------------|---------------|---------|----------|----------|----------|-----------|--------|------------|---------|---------|-------|-----------|---------|---------|---------|-
100 | False | {'id': 86224, 'name': 'The Saint Collection', 'poster_path': '/3dliYexStkZy6NOAH533Y8oduOj.jpg', 'backdrop_path': None} | 1350000 | [{'id': 35, 'name': 'Comedy'}, {'id': 80, 'name': 'Crime'}] | http://www.universalstudiosentertainment.com/lock-stock-and-two-smoking-barrels/ | 100 | tt0120735 | en | Lock, Stock and Two Smoking Barrels | A card sharp and his unwillingly-enlisted friends need to make a lot of cash quick after losing a sketchy poker match. To do this they decide to pull a heist on a small-time gang who happen to be operating out of the flat next door. | 4.60786 | /qV7QaSf7f7yC2lc985zfyOJIAIN.jpg | [{'name': 'Handmade Films Ltd.', 'id': 146}, {'name': 'Summit Entertainment', 'id': 491}] | [{'iso_3166_1': 'GB', 'name': 'United Kingdom'}] | 1998-03-05 | 3897569.0 | 105.0 | [{'iso_639_1': 'en', 'name': 'English'}] | Released | A Disgrace to Criminals Everywhere. | Lock, Stock and Two Smoking Barrels | False | 8 | 1671

`MOVIES_AGGREGATED`
movie_id | popularity | revenue | budget | release_date | genre | production_company | release_year | profit
-----------|--------------|----------|----------|----------------|--------|---------------------------|----------------|------
100 | 4.60786 | 3897569.0 | 1350000 | 1998-03-05 | Comedy | Handmade Films Ltd. | 1998 | 2547569.0

`KEYWORDS_MASTER`
movie_id | keyword_id | keyword
-----------|---------------|---------|
862 | 931 | jealousy

`CREDITS_MASTER`
movie_id | cast | crew
-----------|-------|------|
862 | [{'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}, {'cast_id': 15, 'character': 'Buzz Lightyear (voice)', 'credit_id': '52fe4284c3a36847f8024f99', 'gender': 2, 'id': 12898, 'name': 'Tim Allen', 'order': 1, 'profile_path': '/uX2xVf6pMmPepxnvFWyBtjexzgY.jpg'} | [{\'credit_id\': \'52fe4284c3a36847f8024f49\', \'department\': \'Directing\', \'gender\': 2, \'id\': 7879, \'job\': \'Director\', \'name\': \'John Lasseter\', \'profile_path\': \'/7EdqiNbr4FRjIhKHyPPdFfEEEFG.jpg\'}, {\'credit_id\': \'52fe4284c3a36847f8024f4f\', \'department\': \'Writing\', \'gender\': 2, \'id\': 12891, \'job\': \'Screenplay\', \'name\': \'Joss Whedon\', \'profile_path\': \'/dTiVsuaTVTeGmvkhcyJvKp2A5kr.jpg\'}

`CAST_MASTER`
movie_id | cast_id | character | credit_id | gender | id | name | order | profile_path
-----------|---------|-------------|-----------|----------|----|--------|-------|-------
862 | 14 | Woody (voice) | 52fe4284c3a36847f8024f95 | 2 | 31 | Tom Hanks | 0 | pQFoyx7rp09CJTAb932F2g8Nlho.jpg

`CREW_MASTER`
movie_id | credit_id | department| gender| id| job| name | profile_path
-----------|---------|-------------|-----------|----------|----|--------|-------|------- 
862 | 52fe4284c3a36847f8024f49 | Directing | 2| 7879 | Director | John Lasseter | /7EdqiNbr4FRjIhKHyPPdFfEEEFG.jpg

`RATINGS_MASTER`
movie_id | user_id | rating | timestamp
-----------|----------|--------|--------|
110 | 1 | 3 | 1425941529

`LINKS_MASTER`
movies_id | imdb_id | tmdb_id
------------|-----------|--------
1 | 114709 | 862

### SQL
This section will provide the required SQL queries (using Snowflake syntax and functions) for importing data into tables and retrieving desired aggregate data functions.

##### Data Import
Snowflake can directly integrate with S3, which makes it easy to copy data from S3 files into Snowflake.
First, new file formats will need to be created, one for .csv files and one for .json files
```sql
create or replace file format <schema name>.CSV_HEADER
type = csv
null_if = ('NULL', 'null')
compression = none
skip_header = 1;

create or replace file format <schema name>.JSON_DATA
type = json
trim_space = TRUE
null_if = ()
compression = none;
```
Next, we can create new AWS S3 stages for each file format
```sql
create or replace stage <schema name>.MOVIES_CSV
file_format = <schema name>.CSV_HEADER
url = 's3://com.guild.us-west-2.public-data/project-data/'
credentials = (aws_key_id='*****' aws_secret_key='*****');

create or replace stage <schema name>.MOVIES_JSON
file_format = <schema name>.JSON_DATA
url = 's3://com.guild.us-west-2.public-data/project-data/'
credentials = (aws_key_id='*****' aws_secret_key='*****');
```
Now, we can copy in the .csv files for the unscaled Python script (assuming tables have already been made):
```sql
copy into <schema name>.MOVIES_MASTER
from @MOVIES_CSV/<dated_folder>/movies_cleaned.csv
on_error = continue;

copy into <schema name>.MOVIES_AGGREGATED
from @MOVIES_CSV/<dated_folder>/movies_agg_cleaned.csv
on_error = continue;

copy into <schema name>.KEYWORDS_MASTER
from @MOVIES_CSV/<dated_folder>/keywords_cleaned.csv
on_error = continue;

copy into <schema name>.CREDITS_MASTER
from @MOVIES_CSV/<dated_folder>/credits_cleaned.csv
on_error = continue;

copy into <schema name>CREW_MASTER
from @MOVIES_CSV/<dated_folder>/crew_cleaned.csv
on_error = continue;

copy into <schema name>.CAST_MASTER
from @MOVIES_CSV/<dated_folder>/cast_cleaned.csv
on_error = continue;

copy into <schema name>.LINKS_MASTER
from @MOVIES_CSV/<dated_folder>/links_cleaned.csv
on_error = continue;
```
Similar queries can be done to load in JSON data to table:
```sql
copy into <schema name>.MOVIES_MASTER
from @MOVIES_JSON/<dated_folder>/movies_cleaned.json
on_error = continue;

copy into <schema name>.MOVIES_AGGREGATED
from @MOVIES_JSON/<dated_folder>/movies_aggregated.json
on_error = continue;

copy into <schema name>.KEYWORDS_MASTER
from @MOVIES_JSON/<dated_folder>/keywords.json
on_error = continue;

copy into <schema name>.CREDITS_MASTER
from @MOVIES_JSON/<dated_folder>/credits.json
on_error = continue;

copy into <schema name>CREW_MASTER
from @MOVIES_JSON/<dated_folder>/crew.json
on_error = continue;

copy into <schema name>.CAST_MASTER
from @MOVIES_JSON/<dated_folder>/cast.json
on_error = continue;

copy into <schema name>.LINKS_MASTER
from @MOVIES_JSON/<dated_folder>/links.json
on_error = continue;
```
The `MOVIES_AGGREGATED` table is the one we will be using to answer the following questions:
Production Company Details:
- budget per year
```sql
select production_company, release_year, sum(budget) as total_budget
from (select production_company, movie_id, release_year, budget
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, budget)
group by production_company, release_year
order by production_company, release_year;
```
- revenue per year
```sql
select production_company, release_year, sum(revenue) as total_revenue
from (select production_company, movie_id, release_year, revenue
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, revenue)
group by production_company, release_year
order by production_company, release_year;
```
- profit per year
```sql
select production_company, release_year, sum(profit) as total_profit
from (select production_company, movie_id, release_year, profit
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, profit)
group by production_company, release_year
order by production_company, release_year;
```
- releases by genre per year
```sql
select production_company, genre, release_year, count(distinct movie_id) as release_count
from MOVIES_AGGREGATED
group by production_company, genre, release_year
order by production_company, genre, release_year;
```
- average popularity of produced movies per year
```sql
select production_company, release_year, avg(popularity) as avg_popularity
from (select production_company, movie_id, release_year, popularity
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, popularity)
group by production_company, release_year
order by production_company, release_year;
```
Movie Genre Details:
- most popular genre by year
```sql
select production_company, release_year, sum(budget) as total_budget
from (select production_company, movie_id, release_year, budget
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, budget)
group by production_company, release_year
order by production_company, release_year;
```
- revenue per year
```sql
select production_company, release_year, sum(revenue) as total_revenue
from (select production_company, movie_id, release_year, revenue
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, revenue)
group by production_company, release_year
order by production_company, release_year;
```
- profit per year
```sql
select production_company, release_year, sum(profit) as total_profit
from (select production_company, movie_id, release_year, profit
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, profit)
group by production_company, release_year
order by production_company, release_year;
```
- releases by genre per year
```sql
select production_company, genre, release_year, count(distinct movie_id) as release_count
from MOVIES_AGGREGATED
group by production_company, genre, release_year
order by production_company, genre, release_year;
```
- average popularity of produced movies per year
```sql
select production_company, release_year, avg(popularity) as avg_popularity
from (select production_company, movie_id, release_year, popularity
      from MOVIES_AGGREGATED
      group by production_company, release_year, movie_id, popularity)
group by production_company, release_year
order by production_company, release_year;
```


### Backfilling
How would you go about backfilling 1 year worth of data?

Because the code in this project uses the Boto3 package through Python, it is able to also pull metadata for files dropped in S3. A way of backfilling a year of data would be to create a separate Python file, set the 1-year-previous date as a variable, scan through the files in the bucket matching the desired suffix (e.g. `the-movies-dataset.zip`), and iterate through each file whose upload/create date falls after that 1-year-previous-date. In order to help avoid impact on production flow, a separate Lambda function could be created to run the code on-demand. If Docker/Kubernetes pods are being used, a manual call of the one-time load code could also be performed.
