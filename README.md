# 🕵️‍♂️ Twitter Bot Detection

This repository follows a **standard Python package** layout.

## Project structure

- `src/twitter_bot_detection/`
  - `etl.py`
  - `eda.py`
  - `feature_selection.py`
- `notebooks/example_workflow.ipynb` (single example notebook)
- `datasets/` (data files/placeholders)
- `pyproject.toml`
- `requirements.txt`

## Install

```bash
pip install -e .
pip install -r requirements.txt
```

## Usage

```python
from twitter_bot_detection.etl import make_profile_df, make_tweets_df
from twitter_bot_detection.eda import profile_data_preprocessing
from twitter_bot_detection.feature_selection import backwards_shap_feature_selection
```

Use `notebooks/example_workflow.ipynb` as the reference notebook.
