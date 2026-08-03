import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def profile_data_preprocessing(profile_df, label_df, split_df, neighbor_df, random_seed=42):
    """
    Preprocess the profile data by merging it with label, split, and neighbor dataframes,
    and adding various computed columns.

    Parameters:
    profile_df (pd.DataFrame): DataFrame containing user profile data.
    label_df (pd.DataFrame): DataFrame containing user labels.
    split_df (pd.DataFrame): DataFrame containing data splits.
    neighbor_df (pd.DataFrame): DataFrame containing user neighbor relations.

    Returns:
    pd.DataFrame: Preprocessed DataFrame with additional columns and merged data.
    """

    # Process follow and friend relationships
    follow_df = (
        neighbor_df
        [lambda d: d["relation"].isin(["follow", "friend"])]  # Filter only 'follow' and 'friend' relations
        .groupby(["source_id", "relation"])["target_id"]  # Group by source_id and relation
        .apply(list)  # Aggregate target_id as a list
        .to_frame()  # Convert to DataFrame
        .reset_index()  # Reset index
        .pivot(values="target_id", index="source_id", columns="relation")  # Pivot the DataFrame
        .reset_index()  # Reset index
        .rename(columns={"source_id": "id"})  # Rename source_id to id
    )

    # Process followed relationships
    followed_df = (
        neighbor_df
        [lambda d: d["relation"].isin(["follow"])]  # Filter only 'follow' relations
        .groupby(["target_id", "relation"])["source_id"]  # Group by target_id and relation
        .apply(list)  # Aggregate source_id as a list
        .to_frame()  # Convert to DataFrame
        .reset_index()  # Reset index
        .pivot(values="source_id", index="target_id", columns="relation")  # Pivot the DataFrame
        .reset_index()  # Reset index
        .rename(columns={"target_id": "id"})  # Rename target_id to id
        .rename(columns={"follow": "followed"})  # Rename 'follow' column to 'followed'
    )

    # Merge all dataframes
    merged_df = (
        profile_df
        .merge(label_df, on="id", how="left")  # Merge with label_df on id
        .merge(split_df, on="id", how="left")  # Merge with split_df on id
        .merge(follow_df, on="id", how="left")  # Merge with follow_df on id
        .merge(followed_df, on="id", how="left")  # Merge with followed_df on id
    )

    merged_df["created_at"] = pd.to_datetime(merged_df["created_at"])  # Convert created_at to datetime

    # Use only train split to build reference_date when available to avoid leakage
    train_mask = (
        merged_df["split"].astype(str).str.lower().eq("train")
        if "split" in merged_df.columns
        else pd.Series(False, index=merged_df.index)
    )
    train_reference_date = merged_df.loc[train_mask, "created_at"].max()
    reference_date = (
        train_reference_date
        if pd.notna(train_reference_date)
        else merged_df["created_at"].max()
    )

    rng = np.random.default_rng(random_seed)

    # Add computed columns
    return merged_df.assign(
        followers_count=lambda d: d.public_metrics.str["followers_count"],  # Extract followers_count from public_metrics
        following_count=lambda d: d.public_metrics.str["following_count"],  # Extract following_count from public_metrics
        followers_follow_proportion=lambda d: d["followers_count"] / (d["following_count"] + 1e-3),  # Calculate followers to following proportion
        listed_count=lambda d: d.public_metrics.str["listed_count"],  # Extract listed_count from public_metrics
        tweet_count=lambda d: d.public_metrics.str["tweet_count"],  # Extract tweet_count from public_metrics
        reference_date=reference_date,  # Reference date for tenure computation
        tenure=lambda d: (d["reference_date"] - d["created_at"]).dt.days,  # Calculate tenure in days
        follow_string=lambda d: d["follow"].str.join(" ").fillna(""),  # Join follow list into a string
        followed_string=lambda d: d["followed"].str.join(" ").fillna(""),  # Join followed list into a string
        friend_string=lambda d: d["friend"].str.join(" ").fillna(""),  # Join friend list into a string
        random_number=lambda d: rng.random(len(d))  # Add a reproducible random number column
    )
