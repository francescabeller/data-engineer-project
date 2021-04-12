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
from boto3.s3.transfer import TransferConfig
import json


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


def set_target_prefix(target_path):
    # Create 'target_prefix' with dated folder for reference
    now = datetime.now().strftime("%Y-%m-%dT%H.%MZ")
    target_prefix = '/'.join(target_path.replace('s3://', '').split('/')[1:3]) + '/' + now

    return target_prefix


class MovieDataProcessorScaled:
    __metaclass__ = ABCMeta

    def __init__(
            self, source_path: object = 's3://com.guild.us-west-2.public-data/project-data/the-movies-dataset.zip',
            target_path: object = 's3://com.guild.us-west-2.public-data/project-data/movie_data_output',
    ) -> object:
        self.s3_bucket = None
        self.source_path = source_path
        self.prefix = None
        self.target_path = target_path
        self.target_prefix = None
        self.file_set = None
        self.client = None
        self.chunk_size = 10000
        self.agg_cols = {'adult': 'first', 'belongs_to_collection': 'first', 'budget': 'mean', 'genres': 'first',
                         'homepage': 'first', 'id': 'first', 'imdb_id': 'first', 'original_language': 'first',
                         'original_title': 'first', 'overview': 'first', 'popularity': 'mean', 'poster_path': 'first',
                         'production_companies': 'first', 'production_countries': 'first', 'release_date': 'first',
                         'revenue': 'mean', 'runtime': 'mean', 'spoken_languages': 'first', 'status': 'first',
                         'tagline': 'first', 'title': 'first', 'video': 'first', 'vote_average': 'mean',
                         'vote_count': 'mean'}

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

        # Set S3 source path and prefix
        self.s3_bucket, self.prefix = parse_s3_source_path(self.source_path)

        # Set target S3 prefix
        self.target_prefix = set_target_prefix(self.target_path)

        # Create S3 Boto3 client session if one does not exist
        self.create_s3_client_session()

        # Create in-memory buffer for storing 'the-movies-dataset.zip'
        self.file_set = self.get_s3_objects()

        # Process 'keywords.csv' data
        self.process_keywords_data(self.file_set)
        # Process 'credits.csv' data
        self.process_credits_data(self.file_set)
        # Process 'links.csv' data
        self.process_links_data(self.file_set)
        # Process 'movies_metadata.csv' data
        self.process_movie_data(self.file_set)
        # Generate aggregated dataframe from 'movies_metadata.csv' and upload to S3
        self.create_aggregated_dataframe(self.file_set)

        # Upload log file to S3
        try:
            output_log_path = self.target_prefix + '/movie_data_script.log'
            self.client.upload_file('movie_data_script.log', self.s3_bucket, output_log_path)
            logging.info(f'Successfully uploaded log file to S3.\n')
        except ClientError as e:
            logging.error(f'Error uploading log file to S3: {e}\n')

    def create_aggregated_dataframe(self, fileset):
        """
        Process data from 'movies_metadata.csv', create transformed dataframe for aggregation functions/querying,
        and expand JSON to rows
        :param fileset: ZipFile object containing 'the-movies-dataset.zip' content
        :return: Upload JSON files to S3 and return response
        """

        # Check if 'movies_metadata.csv' is present in S3 directory.
        # If not present, log warning and continue through other files
        # If present, iterate through file in chunks of 10,000
        if 'movies_metadata.csv' not in fileset.namelist():
            logging.warning(f'File movies_metadata.csv not found in S3 directory. Skipping file.\n')
            return
        else:
            for movies_df in pd.read_csv(
                    filepath_or_buffer=fileset.open('movies_metadata.csv'),
                    header=0,
                    low_memory=True,
                    iterator=True,
                    chunksize=self.chunk_size
            ):

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
                # As there are some extraneous strings in the 'genres' column, we will only take those with proper
                # formatting
                agg_df = agg_df[(agg_df['genres'].str.startswith('[')) & (agg_df['genres'].str.endswith(']'))]

                # The 'genres' JSON column is imported as a full string, so we will need to perform some pre-processing
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

                # Upload 'movies_df' to S3
                self.json_upload_to_s3(self.target_path, agg_df, 'movies_aggregated.json')

    def process_movie_data(self, fileset):
        # Check if 'movies_metadata.csv' is present in S3 directory.
        # If not present, log warning and continue through other files
        # If present, iterate through file in chunks of 10,000
        if 'movies_metadata.csv' not in fileset.namelist():
            logging.error(f'Movies metadata dataframe does not contain data. Unable to generate '
                          f'aggregated data table.\n')
            return
        else:
            for movies_df in pd.read_csv(
                    filepath_or_buffer=fileset.open('movies_metadata.csv'),
                    header=0,
                    low_memory=True,
                    iterator=True,
                    chunksize=self.chunk_size
            ):

                # Check if dataframe is empty; if it is, exit function
                if movies_df.empty:
                    logging.warning(f'Movies dataframe does not contain data. Skipping file.\n')
                    return

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

                # Upload 'movies_df' to S3
                self.json_upload_to_s3(self.target_path, movies_df, 'movies_clean.json')

    def process_keywords_data(self, fileset):
        """
        Cleanse data from 'keywords.csv' and expand JSON to rows
        :param fileset: ZipFile object containing 'the-movies-dataset.zip' content
        :return: Upload JSON files to S3 and return response
        """

        # Check if 'keywords.csv' is present in S3 directory.
        # If not present, log warning and continue through other files
        # If present, iterate through file in chunks of 10,000
        if 'keywords.csv' not in fileset.namelist():
            logging.warning(f'File keywords.csv not found in S3 directory. Skipping file.\n')
            return
        else:
            for keywords_df in pd.read_csv(
                    filepath_or_buffer=fileset.open('keywords.csv'),
                    header=0,
                    low_memory=True,
                    iterator=True,
                    chunksize=self.chunk_size
            ):

                # Check if dataframe is empty; if it is, exit function
                if keywords_df.empty:
                    logging.warning(f'Keywords dataframe does not contain data. Skipping file.\n')
                    return

                # Drop any null rows
                keywords_df.dropna(inplace=True)
                # Rename 'id' column to 'movie_id' for easier processing of JSON column
                keywords_df.rename(columns={'id': 'movie_id'}, inplace=True)
                # Keep rows with proper list formatting in 'keywords' str column
                keywords_df = keywords_df[
                    (keywords_df['keywords'].str.startswith('[')) & (keywords_df['keywords'].str.endswith(']'))]

                # The 'keywords' JSON column is imported as a string, so we will need to perform some pre-processing...
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


                # Now, we can perform single or multi-part uploads of the dataframe to S3 in JSON format
                self.json_upload_to_s3(self.target_path, keywords_df, 'keywords_clean.json')

    def process_credits_data(self, fileset):
        """
        Cleanse data from 'credits.csv' and expand JSON to rows
        :param fileset: ZipFile object containing 'the-movies-dataset.zip' content
        :return: Upload JSON files to S3 and return response
        """

        # Check if 'credits.csv' is present in S3 directory.
        # If not present, log warning and continue through other files
        # If present, iterate through file in chunks of 10,000
        if 'credits.csv' not in fileset.namelist():
            logging.warning(f'File credits.csv not found in S3 directory. Skipping file.\n')
            return
        else:
            for credits_df in pd.read_csv(
                    filepath_or_buffer=fileset.open('credits.csv'),
                    header=0,
                    low_memory=True,
                    iterator=True,
                    chunksize=self.chunk_size
            ):

                # Check if dataframe is empty; if it is, exit function
                if credits_df.empty:
                    logging.warning(f'Credits dataframe does not contain data. Skipping file.\n')
                    return

                # Rename 'id' column to 'movie_id' for easier processing of cast/crew JSON columns
                credits_df.rename(columns={'id': 'movie_id'}, inplace=True)
                # Drop any null rows
                credits_df.dropna(inplace=True)
                # Trim whitespace from any string columns
                credits_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

                # Upload full 'credits_df' to S3
                self.json_upload_to_s3(self.target_path, credits_df, 'credits_clean.json')

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

                # Upload 'cast_df' to S3
                self.json_upload_to_s3(self.target_path, cast_df, 'cast_clean.json')

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

                # Upload 'crew_df' to S3
                self.json_upload_to_s3(self.target_path, crew_df, 'crew_clean.json')

    def process_links_data(self, fileset):
        """
        Cleanse data from 'links.csv' and expand JSON to rows
        :param fileset: ZipFile object containing 'the-movies-dataset.zip' content
        :return: Upload JSON files to S3 and return response
        """

        # Check if 'links.csv' is present in S3 directory.
        # If not present, log warning and continue through other files
        # If present, iterate through file in chunks of 10,000
        if 'links.csv' not in fileset.namelist():
            logging.warning(f'File links.csv not found in S3 directory. Skipping file.\n')
            return
        else:
            for links_df in pd.read_csv(
                    filepath_or_buffer=fileset.open('links.csv'),
                    header=0,
                    low_memory=True,
                    iterator=True,
                    chunksize=self.chunk_size
            ):

                # Check if dataframe is empty; if it is, exit function
                if links_df.empty:
                    logging.warning(f'Links dataframe does not contain data. Skipping file.\n')
                    return

                # Convert 'tmdbId' to type int
                links_df['tmdbId'] = links_df['tmdbId'].astype(int)

                # Rename columns to match standard
                links_df.rename(columns={'movieId': 'movie_id', 'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'}, inplace=True)

                # Upload 'links_df' to S3
                self.json_upload_to_s3(self.target_path, links_df, 'links_clean.json')

    def process_ratings_data(self, fileset):
        """
        Cleanse data from 'links.csv' and expand JSON to rows
        :param fileset: ZipFile object containing 'the-movies-dataset.zip' content
        :return: Upload JSON files to S3 and return response
        """

        # Check if 'links.csv' is present in S3 directory.
        # If not present, log warning and continue through other files
        # If present, iterate through file in chunks of 10,000
        if 'ratings.csv' not in fileset.namelist():
            logging.warning(f'File ratings.csv not found in S3 directory. Skipping file.\n')
            return
        else:
            for ratings_df in pd.read_csv(
                    filepath_or_buffer=fileset.open('ratings.csv'),
                    header=0,
                    low_memory=True,
                    iterator=True,
                    chunksize=self.chunk_size
            ):

                # Check if dataframe is empty; if it is, exit function
                if ratings_df.empty:
                    logging.warning(f'Ratings dataframe does not contain data. Skipping file.\n')
                    return

                # Convert 'rating' to type int
                ratings_df['rating'] = ratings_df['rating'].astype(int)

                # Trim whitespace from any string columns
                ratings_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

                # Rename columns to match standard
                ratings_df.rename(columns={'movieId': 'movie_id', 'userId': 'user_id'}, inplace=True)

                # Upload 'links_df' to S3
                self.json_upload_to_s3(self.target_path, ratings_df, 'ratings_clean.json')

    def create_s3_client_session(self):
        """
        Initialize Boto3 S3 client session if none currently exists
        :return: Boto3 S3 client session
        """

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
            logging.error(f'Unable to establish boto3 S3 connection.\n')

    def get_s3_objects(self):

        # Iterate through S3 directory objects and check for file with today's date
        objects = self.client.list_objects(Bucket=self.s3_bucket, Prefix=self.prefix)
        for o in objects["Contents"]:
            if o["LastModified"].date() == datetime.today().strftime('%Y-%m-%d'):
                # If the file was uploaded today, extract the zip file object
                obj = self.client.get_object(Bucket=self.s3_bucket, Key=self.prefix)
                logging.info(f'File found with upload date of today. Continuing with script processes.\n')
                # Read zip file to buffer for use with pandas
                buffer = io.BytesIO(obj['Body'].read())
                # Set zipfile variable
                z = zipfile.ZipFile(buffer)

                return z

            else:
                logging.info(f'No file uploaded today. Ending script run.\n')
                return

    def json_upload_to_s3(self, target_path, dataframe, file_name):
        """
        Function to upload a block up uploading records. Based on our default chunking size)
        :param target_path: str(output S3 path)
        :param dataframe: pandas DataFrame object
        :param file_name: str(name of output JSON file)
        :return: None
        """

        # Create 'target_prefix' with dated folder for reference
        now = datetime.now().strftime("%Y-%m-%dT%H.%MZ")
        target_prefix = '/'.join(target_path.replace('s3://', '').split('/')[1:3]) + '/' + now

        # Create JSON buffer and set to BytesIO object with converted pandas DataFrame
        json_buffer = io.BytesIO(str.encode(json.dumps(dataframe.to_dict(orient='records'))[1:-1], 'utf-8'))

        # Set the output file name/path
        file_path = target_prefix + '/' + file_name

        # In order to perform multi-part uploading for large files to S3, we need to use
        # boto3's 'TransferConfig' and set our max data size (20mb) per upload part/chunk
        mb = 1024 ** 2
        config = TransferConfig(multipart_threshold=20 * mb, max_concurrency=80,
                                multipart_chunksize=20 * mb, use_threads=True)

        # Now we can use the 'upload_fileob' with our 'TransferConfig' to upload the large JSON pandas conversion
        # to S3
        try:
            response = self.client.upload_fileobj(Fileobj=json_buffer,
                                                  Bucket=self.s3_bucket,
                                                  Key=file_path,
                                                  ExtraArgs={'ContentType': 'application/json'},
                                                  Config=config)
            logging.info(f'Successfully uploaded {file_name} to S3.\n')
        except ClientError as e:
            logging.error(f'Error uploading {file_name} to S3: {e}\n')


if __name__ == '__main__':
    MovieDataProcessorScaled.movie_processor()
