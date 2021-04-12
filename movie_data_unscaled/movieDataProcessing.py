import pandas as pd
import numpy as np
import boto3
import logging
from abc import ABCMeta
import zipfile
import io
from params import Environment as env
from datetime import datetime
from botocore.exceptions import ClientError


def parse_s3_source_path(source_path):
    """
    Parses an S3 path in the s3://<bucket_name>/prefix format
    :param source_path: str(path to parse in s3://<bucket_name>/prefix format)
    :return: str(bucket name from the path), str(prefix to the zip file)
    """
    source_path = source_path.replace('s3://', '').split('/')
    bucket_name = source_path[0]
    prefix = '/'.join(source_path[1:])
    return bucket_name, prefix


class MovieDataProcessor:
    __metaclass__ = ABCMeta

    def __init__(
            self, dataframe_dict: object = None, s3_bucket: object = None,
            source_path: object = 's3://com.guild.us-west-2.public-data/project-data/the-movies-dataset.zip',
            target_path: object = 's3://com.guild.us-west-2.public-data/project-data/movie_data_output',
    ) -> object:
        self.s3_bucket = s3_bucket
        self.source_path = source_path
        self.target_path = target_path
        self.dataframe_dict = dataframe_dict
        self.client = None
        self.agg_cols = {'adult': 'first', 'belongs_to_collection': 'first', 'budget': 'mean', 'genres': 'first',
                         'homepage': 'first', 'id': 'first', 'imdb_id': 'first', 'original_language': 'first',
                         'original_title': 'first', 'overview': 'first', 'popularity': 'mean', 'poster_path': 'first',
                         'production_companies': 'first', 'production_countries': 'first', 'release_date': 'first',
                         'revenue': 'mean', 'runtime': 'mean', 'spoken_languages': 'first', 'status': 'first',
                         'tagline': 'first', 'title': 'first', 'video': 'first', 'vote_average': 'mean',
                         'vote_count': 'mean'}
        self.output_file_dict = {}

    def movie_processor(self):
        """
        Main ETL
        :return:
        """

        # Initiate log file and add script start-time
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            filename='movie_data_script.log',
                            level=logging.INFO)
        logging.info(f'Movie data script start time: {datetime.now()}')

        # Generate dictionary of dataframes from zip files
        self.dataframe_dict = self.import_files()

        # Cleanse 'keywords.csv' data
        self.dataframe_dict = self.cleanse_keywords_data(self.dataframe_dict)
        # Cleanse 'links.csv' data - simply need to convert 'tmdbId' column from float64 to int
        links_df = self.dataframe_dict['links_df']
        links_df['tmdbId'] = links_df['tmdbId'].astype(int)
        # Rename columns to match standard
        links_df.rename(columns={'movieId': 'movie_id', 'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'}, inplace=True)
        self.dataframe_dict['links_df'] = links_df
        # Cleanse 'credits.csv'
        self.dataframe_dict = self.cleanse_credits_data(self.dataframe_dict)
        # Cleanse 'ratings.csv' data - just need to convert 'rating' from float64 to int
        ratings_df = self.dataframe_dict['ratings_df']
        ratings_df['rating'] = ratings_df['rating'].astype(int)
        # Rename columns to match standard
        ratings_df.rename(columns={'movieId': 'movie_id', 'userId': 'user_id'}, inplace=True)
        self.dataframe_dict['ratings_df'] = ratings_df

        # Create dataframe for aggregate functions
        self.dataframe_dict = self.create_base_dataframe(self.cleanse_movie_data(self.dataframe_dict))

        # Upload new .csv and log files to S3
        self.upload_files_to_s3(self.dataframe_dict, self.target_path, self.s3_bucket)

    def import_files(self):
        """
        Get files from a given S3 path
        :return: dict{dataframe name: data}
        """

        # Create dictionary of files and dataframe names
        # Note: we will not be importing 'ratings_small.csv' or 'links_small.csv' as they are simply subsets of the full
        # data in 'ratings.csv' and 'links.csv'
        file_dict = {'movies_df': 'movies_metadata.csv',
                     'ratings_df': 'ratings.csv',
                     'credits_df': 'credits.csv',
                     'keywords_df': 'keywords.csv',
                     'links_df': 'links.csv'}

        # Initialize Boto3 S3 client session if none currently exists
        try:
            if not self.client:
                self.client = boto3.client(
                    's3',
                    region_name=env.REGION,
                    aws_access_key_id=env.AWS_ACCESS_KEY,
                    aws_secret_access_key=env.AWS_SECRET_KEY
                )
            logging.info(f'S3 connection successful.\n')
        except Exception as e:
            logging.error(f'Unable to establish boto3 S3 connection. Aborting script. \n')
            return

        # Extract bucket name and file prefix from source path
        self.s3_bucket, prefix = parse_s3_source_path(self.source_path)

        # Iterate through S3 directory objects and check for file with today's date
        objects = self.client.list_objects(Bucket=self.s3_bucket, Prefix=prefix)
        for o in objects["Contents"]:
            if o["LastModified"].date() == datetime.today().strftime('%Y-%m-%d'):
                # If the file was uploaded today, extract the zip file object
                obj = self.client.get_object(Bucket=self.s3_bucket, Key=prefix)
                logging.info(f'File found with upload date of today. Continuing with script processes.\n')
            else:
                logging.info(f'No file uploaded today. Ending script run.\n')
                return

        # Get Bucket Object
        obj = self.client.get_object(Bucket=self.s3_bucket, Key=prefix)
        # Read zip file to buffer for use with pandas
        buffer = io.BytesIO(obj['Body'].read())
        # Set zipfile variable
        z = zipfile.ZipFile(buffer)
        # Iterate through 'file_dict' to generate dataframes from files in zip and add to dictionary
        df_dict = {}
        expected_count = len(file_dict)
        actual_count = 0
        for key, value in file_dict.items():
            # If 'movies_metadata.csv' is not present in S3 directory, aggregate table cannot be made and script will
            # be aborted
            if value not in z.namelist() and value == 'movies_metadata.csv':
                logging.error(f'Necessary movies_metadata.csv file not found, aborting script.\n')
                return
            # If files other than 'movies_metadata.csv' are not in the S3 directory, log a warning and continue through
            # script
            elif value not in z.namelist() and value != 'movies_metadata.csv':
                logging.warning(f'File {value} not found in S3 directory. Skipping file.\n')
                continue
            else:
                vars()[key] = pd.read_csv(z.open(value))
                df_dict[key] = vars()[key]
                actual_count += 1
        # Check if all 5 expected files were imported
        if actual_count == expected_count:
            logging.info(f'All S3 files successfully imported from metadata_movies.zip.\n')
        else:
            logging.warning(f'Number of files imported does not match expected file count (5).\n')

        return df_dict

    @staticmethod
    def create_base_dataframe(df_dict):
        """
        Generate a master dataframe to contain data from all files in metadata_movies.zip
        :param df_dict: dict containing dataframes for each of the files in metadata_movies.zip
        :return: dataframe dictionary with new
        """

        # Extract movies dataframe from dataframe list
        movies_df = df_dict['movies_df']

        # Check if dataframe is empty
        if movies_df.empty:
            logging.error(f'Movies metadata dataframe does not contain data. Unable to generate '
                          f'aggregated data table.\n')
            return

        # Process movies dataframe to extract genre information
        agg_df = movies_df[['id', 'genres', 'production_companies', 'popularity', 'revenue', 'budget', 'release_date']]
        # Rename 'id' column to 'movie_id' for easier processing of genres/production_companies JSON columns
        agg_df.rename(columns={'id': 'movie_id'}, inplace=True)
        # Drop any null rows
        agg_df.dropna(inplace=True)
        # As there are some extraneous strings in the 'genres' column, we will only take those with proper formatting
        agg_df = agg_df[(agg_df['genres'].str.startswith('[')) & (agg_df['genres'].str.endswith(']'))]

        # The 'genres' JSON column is imported as a full string, so we will need to perform some pre-processing...
        # Create placeholder list of current cell values in 'genres'
        r = list(agg_df['genres'])
        # Iterate through 'r' list and extract dictionary objects
        d_list = []
        for i in range(0, len(r)):
            d_list.append(list(eval(r[i])))
        # Set the 'genres' column in our base dataframe to the reformatted 'd_list'
        agg_df['genres'] = d_list

        # Next, we need to create a new row for each dictionary value in a record's 'genres' cell value list.
        # To do this, we will create a temp DataFrame and use numpy/pandas to "explode" the dictionary lists
        temp = pd.DataFrame({
            col: np.repeat(agg_df[col].values, agg_df['genres'].str.len())
            for col in agg_df.columns.drop('genres')}
        ).assign(**{'genres': np.concatenate(agg_df['genres'].values)})[agg_df.columns]

        # Now we will simply recreate the base DataFrame by concatenating our temp DataFrame
        agg_df = pd.concat([temp.drop(['genres'], axis=1), temp['genres'].apply(pd.Series)], axis=1)
        # Remove the 'id' column as it is not needed
        del agg_df['id']
        # Rename the extracted 'name' column we got from the 'genres' JSON
        agg_df.rename(columns={'name': 'genre'}, inplace=True)

        # We need to perform the same functions to reformat and expand the 'production_companies' JSON column
        agg_df = agg_df[
            (agg_df['production_companies'].str.startswith('[')) & (agg_df['production_companies'].str.endswith(']'))]

        r = list(agg_df['production_companies'])
        d_list = []
        for i in range(0, len(r)):
            d_list.append(list(eval(r[i])))
        agg_df['production_companies'] = d_list

        temp = pd.DataFrame({
            col: np.repeat(agg_df[col].values, agg_df['production_companies'].str.len())
            for col in agg_df.columns.drop('production_companies')}
        ).assign(**{'production_companies': np.concatenate(agg_df['production_companies'].values)})[agg_df.columns]

        agg_df = pd.concat(
            [temp.drop(['production_companies'], axis=1), temp['production_companies'].apply(pd.Series)], axis=1)
        del agg_df['id']
        agg_df.rename(columns={'name': 'production_company'}, inplace=True)

        # Create 'release_year' column from 'release_date' for easier querying
        agg_df['release_year'] = agg_df.release_date.str[:4]

        # Create 'profit' column by subtracting 'budget' from 'revenue'
        agg_df['profit'] = agg_df['revenue'] - agg_df['budget']

        # Trim whitespace from any string columns
        agg_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Add new aggregate dataframe to 'df_dict'
        df_dict['movies_agg_df'] = agg_df

        # We should now have a full base DataFrame with a separate row for each genre/production_company combination
        return df_dict

    def cleanse_movie_data(self, df_dict):
        """
        Merge duplicate records in 'movies_metadata.csv' dataframe based on 'movie_id', and aggregate numeric columns
        to use the mean/average of the duplicate records
        :param df_dict: dict containing dataframes for each of the files in the-movies-dataset.zip
        :return: dataframe dictionary with cleansed 'movies_metadata.csv' data
        """

        # Pull 'movies_metadata.csv' dataframe from 'df_dict'
        movies_df = df_dict['movies_df']

        # Fill in null values for number columns with 0
        movies_df.fillna({'budget': 0, 'popularity': 0, 'revenue': 0, 'vote_count': 0}, inplace=True)
        # Remove any rows where the value of 'budget' is not numeric
        movies_df = movies_df[(movies_df['budget'].str.isnumeric())]

        # In order to aggregate columns to get mean, we need to make sure they are int/float type
        # Set 'budget' and 'vote_count' to integers
        movies_df[['budget', 'vote_count']] = movies_df[['budget', 'vote_count']].astype(int)
        # Set 'popularity' to float
        movies_df['popularity'] = movies_df['popularity'].astype(float)

        # Using the 'self.agg_cols' dictionary, we can group the 'movies_df' dataframe by 'id' and
        # get the mean for our numeric columns
        movies_df = movies_df.groupby('id', as_index=False).agg(self.agg_cols)
        # Round originally integer columns that may have had decimals added during aggregation
        movies_df[['revenue', 'runtime', 'vote_average', 'vote_count']] = movies_df[
            ['revenue', 'runtime', 'vote_average', 'vote_count']].round()

        # Trim whitespace from any string columns
        movies_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Edit 'df_dict' to have cleansed 'movies_df' data
        df_dict['movies_df'] = movies_df

        return df_dict

    @staticmethod
    def cleanse_keywords_data(df_dict):
        """
        Cleanse data from 'keywords.csv' and expand JSON to rows
        :param df_dict: dict containing dataframes for each of the files in the-movies-dataset.zip
        :return: dataframe dictionary with cleansed 'keywords.csv' data
        """
        # Set keywords dataframe
        keywords_df = df_dict['keywords_df']

        # Check if dataframe is empty; if it is, delete 'keywords_df' from 'df_dict', log error, and return 'df_dict'
        if keywords_df.empty:
            logging.warning(f'Keywords dataframe does not contain data. Skipping file.\n')
            del df_dict['keywords_df']
            return df_dict

        # Drop any null rows
        keywords_df.dropna(inplace=True)
        # Rename 'id' column to 'movie_id' for easier processing of JSON column
        keywords_df.rename(columns={'id': 'movie_id'}, inplace=True)
        # Keep rows with proper list formatting in 'keywords' str column
        keywords_df = keywords_df[
            (keywords_df['keywords'].str.startswith('[')) & (keywords_df['keywords'].str.endswith(']'))]

        # The 'keywords' JSON column is imported as a full string, so we will need to perform some pre-processing...
        # Create placeholder list of current cell values in 'keywords'
        r = list(keywords_df['keywords'])
        # Iterate through 'r' list and extract dictionary objects
        d_list = []
        for i in range(0, len(r)):
            d_list.append(list(eval(r[i])))
        # Set the 'genres' column in our base dataframe to the reformatted 'd_list'
        keywords_df['keywords'] = d_list

        # Next, we need to create a new row for each dictionary value in a record's 'keywords' cell value list.
        # To do this, we will create a temp DataFrame and use numpy/pandas to "explode" the dictionary lists
        temp = pd.DataFrame({
            col: np.repeat(keywords_df[col].values, keywords_df['keywords'].str.len())
            for col in keywords_df.columns.drop('keywords')}
        ).assign(**{'keywords': np.concatenate(keywords_df['keywords'].values)})[keywords_df.columns]

        # Recreate 'keywords_df' by concatenating our temp DataFrame
        keywords_df = pd.concat([temp.drop(['keywords'], axis=1), temp['keywords'].apply(pd.Series)], axis=1)

        # Rename 'id' and 'name' columns from expanded 'keywords' JSON
        keywords_df.rename(columns={'id': 'keyword_id', 'name': 'keyword'}, inplace=True)

        # Trim whitespace from any string columns
        keywords_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Edit 'df_dict' with clean data
        df_dict['keywords_df'] = keywords_df

        return df_dict

    @staticmethod
    def cleanse_credits_data(df_dict):
        """
        Create separate 'cast' and 'crew' dataframes, cleanse data, and add to 'df_dict'
        :param df_dict: dict containing dataframes for each of the files in the-movies-dataset.zip
        :return: dataframe dictionary with new, cleansed cast and crew data
        """

        # Extract movies dataframe from dataframe list
        credits_df = df_dict['credits_df']

        # Check if dataframe is empty; if it is, delete 'credits_df' from 'df_dict', log error, and return 'df_dict'
        if credits_df.empty:
            logging.warning(f'Credits dataframe does not contain data. Skipping file.\n')
            del df_dict['credits_df']
            return df_dict

        # Rename 'id' column to 'movie_id' for easier processing of cast/crew JSON columns
        credits_df.rename(columns={'id': 'movie_id'}, inplace=True)
        # Drop any null rows
        credits_df.dropna(inplace=True)
        # Trim whitespace from any string columns
        credits_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # Replace 'credits_df' in 'df_dict' with cleansed data
        df_dict['credits_df'] = credits_df

        # Create 'cast' dataframe and cleanse

        cast_df = credits_df[['movie_id', 'cast']]
        # Keep rows with proper list formatting in 'cast' str column
        cast_df = cast_df[(cast_df['cast'].str.startswith('[')) & (cast_df['cast'].str.endswith(']'))]

        # The 'cast' JSON column is imported as a full string, so we will need to perform some pre-processing...
        # Create placeholder list of current cell values in 'cast'
        r = list(cast_df['cast'])
        # Iterate through 'r' list and extract dictionary objects
        d_list = []
        for i in range(0, len(r)):
            d_list.append(list(eval(r[i])))
        # Set the 'cast' column in our base dataframe to the reformatted 'd_list'
        cast_df['cast'] = d_list

        # Next, we need to create a new row for each dictionary value in a record's 'cast' cell value list.
        # To do this, we will create a temp DataFrame and use numpy/pandas to "explode" the dictionary lists
        temp = pd.DataFrame({
            col: np.repeat(cast_df[col].values, cast_df['cast'].str.len())
            for col in cast_df.columns.drop('cast')}
        ).assign(**{'cast': np.concatenate(cast_df['cast'].values)})[cast_df.columns]

        # Recreate 'cast_df' by concatenating our temp DataFrame
        cast_df = pd.concat([temp.drop(['cast'], axis=1), temp['cast'].apply(pd.Series)], axis=1)

        # Add 'cast_df' to 'df_dict'
        df_dict['cast_df'] = cast_df

        # Create 'crew' dataframe and cleanse

        crew_df = credits_df[['movie_id', 'crew']]
        # Keep rows with proper list formatting in 'crew' str column
        crew_df = crew_df[(crew_df['crew'].str.startswith('[')) & (crew_df['crew'].str.endswith(']'))]

        # The 'crew' JSON column is imported as a full string, so we will need to perform some pre-processing...
        # Create placeholder list of current cell values in 'crew'
        r = list(crew_df['crew'])
        # Iterate through 'r' list and extract dictionary objects
        d_list = []
        for i in range(0, len(r)):
            d_list.append(list(eval(r[i])))
        # Set the 'crew' column in our base dataframe to the reformatted 'd_list'
        crew_df['crew'] = d_list

        # Next, we need to create a new row for each dictionary value in a record's 'crew' cell value list.
        # To do this, we will create a temp DataFrame and use numpy/pandas to "explode" the dictionary lists
        temp = pd.DataFrame({
            col: np.repeat(crew_df[col].values, crew_df['crew'].str.len())
            for col in crew_df.columns.drop('crew')}
        ).assign(**{'crew': np.concatenate(crew_df['crew'].values)})[crew_df.columns]

        # Recreate 'crew_df' by concatenating our temp DataFrame
        crew_df = pd.concat([temp.drop(['crew'], axis=1), temp['crew'].apply(pd.Series)], axis=1)

        # Add 'crew_df' to 'df_dict'
        df_dict['crew_df'] = crew_df

        return df_dict

    def upload_files_to_s3(self, df_dict, target_path, bucket):
        """
        Upload cleansed dataframes as .csv files to S3
        :param df_dict: dict containing dataframes for each of the files in the-movies-dataset.zip
        :param target_path: str(S3 output path)
        :param bucket: str(S3 bucket name)
        :return:
        """

        # Create dict to hold output file names and data
        for key, value in df_dict.items():
            # Replace 'df' with 'clean.csv' for file names
            temp_str = key.replace('df', 'clean.csv')
            # Append new file name and data to output file dict
            self.output_file_dict[temp_str] = value

        # Create 'target_prefix' with dated folder for reference
        now = datetime.now().strftime("%Y-%m-%dT%H.%MZ")
        target_prefix = '/'.join(target_path.replace('s3://', '').split('/')[1:3]) + '/' + now

        # Iterate through output file dictionary and upload .csv files to S3 output path
        for key, value in self.output_file_dict.items():
            csv_buffer = io.StringIO()
            temp_df = value
            temp_df.to_csv(csv_buffer)
            file_name = target_prefix + '/' + key
            try:
                response = self.client.put_object(Bucket=bucket, Key=file_name, Body=csv_buffer.getvalue())
                logging.info(f'Successfully uploaded {key} to S3.\n')
            except ClientError as e:
                logging.error(f'Error uploading {key} to S3: {e}\n')

        # Upload log file to S3 directory
        try:
            output_log_path = target_prefix + '/movie_data_script.log'
            self.client.upload_file('movie_data_script.log', bucket, output_log_path)
            logging.info(f'Successfully uploaded log file to S3.\n')
        except ClientError as e:
            logging.error(f'Error uploading log file to S3: {e}\n')


