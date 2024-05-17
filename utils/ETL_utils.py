import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

#############################################################
# 1_ETL - Initial preprocessing
#############################################################

# List of columns used to select specific data from dataframe
USECOLS = [
    'id',
    'id_str',
    'name',
    'screen_name',
    'location',
    'profile_location',
    'description',
    # 'url',
    # 'entities',
    'protected',
    'followers_count',
    'friends_count',
    'listed_count',
    'created_at',
    'favourites_count',
    'utc_offset',
    'time_zone',
    'geo_enabled',
    'verified',
    'statuses_count',
    'lang',
    'contributors_enabled',
    'is_translator',
    'is_translation_enabled',
    'profile_background_color',
    'profile_background_image_url',
    'profile_background_image_url_https',
    'profile_background_tile',
    'profile_image_url',
    'profile_image_url_https',
    'profile_link_color',
    'profile_sidebar_border_color',
    'profile_sidebar_fill_color',
    'profile_text_color',
    'profile_use_background_image',
    'has_extended_profile',
    'default_profile',
    'default_profile_image']

import pandas as pd

def make_tweets_df(json_path):
    """
    Loads a JSON dataset of tweets, extracts the 'tweet' and 'ID' columns, and
    transforms the 'tweet' column from a list of tweets to individual rows.

    Parameters:
        json_path (str): The file path to the JSON containing tweet data.
        
    Returns:
        DataFrame: A pandas DataFrame with 'tweet' and 'ID' columns,
                   where 'tweet' is an exploded column from lists of tweets.
    """
    df = pd.read_json(json_path)
    return df[['tweet', 'ID']].explode('tweet')


def _parse_profile_url_json(df, url_attribute="expanded_url"):
    """
    Private helper to extract URLs from a DataFrame containing profile data in JSON format.
    
    Parameters:
        df (DataFrame): A pandas DataFrame containing profile data with an 'entities' column.
        url_attribute (str): The JSON key where the expanded URL can be found.
        
    Returns:
        Series: A pandas Series containing the expanded URLs.
    """
    return df['entities'].apply(lambda entities: entities.get(url_attribute, '') if isinstance(entities, dict) else '')


def make_profile_df(json_path, usecols=USECOLS):
    """
    Loads a JSON dataset of user profiles, extracts specific profile data,
    and returns a DataFrame with specified columns.

    Parameters:
        json_path (str): The file path to the JSON containing user profile data.
        usecols (list of str): List of columns to include in the final DataFrame.
        
    Returns:
        DataFrame: A pandas DataFrame with specified profile data columns.
    """
    df = pd.read_json(json_path)
    profiles_df = pd.DataFrame(df['profile'].values.tolist())
    profiles_df['displayed_urls'] = _parse_profile_url_json(profiles_df)
    profiles_df['domain'] = df['domain']
    return profiles_df[usecols]


def create_id_domain_df(json_path):
    """
    Reads a JSON file into a DataFrame and creates a DataFrame with 'ID' and 'domain'
    columns, where 'domain' is a list turned into a comma-separated string.

    Parameters:
        json_path (str): The file path to the JSON data.
        
    Returns:
        DataFrame: A pandas DataFrame with 'ID' and 'domain' columns.
    """
    df = pd.read_json(json_path)
    df_id_domain = df[['ID', 'domain']].copy()
    df_id_domain['domain'] = df_id_domain['domain'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else x
    )
    return df_id_domain


def create_id_label_df(json_path):
    """
    Reads a JSON file into a DataFrame and extracts 'ID' and 'label' columns.

    Parameters:
        json_path (str): The file path to the JSON data.
        
    Returns:
        DataFrame: A pandas DataFrame with 'ID' and 'label' columns.
    """
    df = pd.read_json(json_path)
    return df[['ID', 'label']].copy()


def create_id_neighbor_df(json_path):
    """
    Reads a JSON file into a DataFrame and extracts 'ID', 'followers', and 'following' columns.
    It handles both dictionary and NoneType entries within the 'neighbor' column, assigning
    None to 'followers' and 'following' when appropriate.

    Parameters:
        json_path (str): The file path to the JSON data.
        
    Returns:
        DataFrame: A pandas DataFrame with 'ID', 'followers', and 'following' columns, where
                   'followers' and 'following' can be None if the 'neighbor' entry is NoneType.
    """
    df = pd.read_json(json_path)
    
    # Function to extract followers and following, accounting for NoneType
    def extract_neighbors(neighbor_data):
        if isinstance(neighbor_data, dict):
            return neighbor_data.get('follower'), neighbor_data.get('following')
        else:
            return None, None  # Return None if neighbor_data is NoneType
    
    # Apply the function to the 'neighbor' column
    df[['followers', 'following']] = df.apply(
        lambda row: extract_neighbors(row['neighbor']), axis=1, result_type="expand"
    )
    
    # Select the columns 'ID', 'followers', and 'following'
    df_id_neighbor = df[['ID', 'followers', 'following']]
    
    return df_id_neighbor