 # 🕵️‍♂️ Twitter Bot Detection

 This repository now follows a **library-first** structure.
 Core pipeline logic lives in a Python package, and only one notebook is kept as usage example.

 ## Project structure

 - `lib/twitter_bot_detection/`
   - `etl.py`
   - `eda.py`
   - `feature_selection.py`
 - `notebooks/example_workflow.ipynb` (single example notebook)

 ## Install

 ```bash
 pip install -e .
 ```

 ## Usage

 Import from the package:

 ```python
 from twitter_bot_detection.etl import make_profile_df, make_tweets_df
 from twitter_bot_detection.eda import profile_data_preprocessing
 from twitter_bot_detection.feature_selection import backwards_shap_feature_selection
 ```

 Use `notebooks/example_workflow.ipynb` as the reference notebook for interactive exploration.
