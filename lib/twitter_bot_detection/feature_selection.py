from sklearn.base import clone
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm
from toolz import curry
from sklearn.metrics import roc_auc_score

###################################################################################################################################
#The provided code implements a set of functions for evaluating the performance of machine learning models using bootstrapping to estimate metrics and their confidence intervals. It also includes a method for feature selection based on SHAP values.
###################################################################################################################################


@curry
def fast_metric_with_ci_(
    data, 
    *, 
    n_samples=100, 
    ci_level=0.95,
    prediction='prediction', 
    target='target', 
    weight='weight', 
    metric_fn=roc_auc_score
):
    """
    This function calculates a specified metric (default is ROC AUC) along with its confidence intervals (ci) using bootstrapping. It takes in a DataFrame data containing predictions and target values, and other parameters such as the number of bootstrap samples (n_samples) and the confidence interval level (ci_level). The metric is computed by rounding predictions, grouping by prediction, target, and weight, and then applying the metric function.

    Parameters:
    data (pd.DataFrame): The data containing predictions and targets.
    n_samples (int): Number of bootstrap samples.
    ci_level (float): Confidence interval level.
    prediction (str): Column name for predictions.
    target (str): Column name for target values.
    weight (str): Column name for weights.
    metric_fn (callable): Metric function to evaluate.

    Returns:
    pd.Series: Estimated metric and confidence intervals.
    """
    data = data.assign(weight__=lambda df: df[weight] if weight is not None else 1)
    summary = (
        data
        .assign(prediction=lambda df: (1000 * df[prediction]).round())
        .groupby(["weight__", 'prediction', target])
        .size().to_frame("sample_size")
        .reset_index()
    )

    estimate = (
        summary
        .assign(weight__=lambda df: df["weight__"] * df['sample_size'])
        .pipe(lambda df: metric_fn(df[target], df['prediction'], sample_weight=df['weight__']))
    )

    bs_values = [
        summary
        .assign(weight__=lambda df: df["weight__"] * np.random.poisson(df['sample_size']))
        .pipe(lambda df: metric_fn(df[target], df['prediction'], sample_weight=df['weight__']))
    for _ in range(n_samples)]

    lo, hi = bootstrap_ci(estimate, bs_values, ci_level=ci_level)

    return pd.Series(dict(
        estimate=estimate,
        ci_upper=hi,
        ci_lower=lo,
        model=prediction
    ))


def bootstrap_ci(sample_estimate, bootstrap_estimates, ci_level=0.95):
    """
    Calculates bootstrap confidence intervals for a given sample estimate. It takes the sample estimate, a list of bootstrap estimates, and the confidence interval level, and returns the lower and upper bounds of the confidence interval.

    Parameters:
    sample_estimate (float): The sample estimate.
    bootstrap_estimates (list): List of bootstrap estimates.
    ci_level (float): Confidence interval level.

    Returns:
    tuple: Lower and upper bounds of the confidence interval.
    """
    lo = 2 * sample_estimate - np.quantile(bootstrap_estimates, (1 + ci_level) / 2)
    hi = 2 * sample_estimate - np.quantile(bootstrap_estimates, (1 - ci_level) / 2)
    return lo, hi


@curry
def fast_delta_metric_with_ci_(
    data, 
    baseline, 
    challenger, 
    *, 
    n_samples=100, 
    ci_level=0.95,
    target='target', 
    weight='weight', 
    metric_fn=roc_auc_score
):
    """
    This function calculates the difference in a specified metric between two sets of predictions (baseline and challenger) along with confidence intervals using bootstrapping. It uses the same process as fast_metric_with_ci_ but calculates the metric difference (delta) instead of the metric itself.

    Parameters:
    data (pd.DataFrame): The data containing predictions and targets.
    baseline (str): Column name for baseline predictions.
    challenger (str): Column name for challenger predictions.
    n_samples (int): Number of bootstrap samples.
    ci_level (float): Confidence interval level.
    target (str): Column name for target values.
    weight (str): Column name for weights.
    metric_fn (callable): Metric function to evaluate.

    Returns:
    pd.Series: Estimated delta metric and confidence intervals.
    """
    data = data.assign(weight__=lambda df: df[weight] if weight is not None else 1)

    summary = (
        data
        .assign(**{
            baseline: lambda df: (1000 * df[baseline]).round(),
            challenger: lambda df: (1000 * df[challenger]).round(),
        })
        .groupby(["weight__", baseline, challenger, target])
        .size().to_frame("sample_size")
        .reset_index()
    )

    def delta_auc(df):
        challenger_auc = metric_fn(df[target], df[challenger], sample_weight=df['weight__'])
        baseline_auc = metric_fn(df[target], df[baseline], sample_weight=df['weight__'])
        return challenger_auc - baseline_auc

    estimate = (
        summary
        .assign(weight__=lambda df: df["weight__"] * df['sample_size'])
        .pipe(delta_auc)
    )

    bs_values = [
        summary
        .assign(weight__=lambda df: df["weight__"] * np.random.poisson(df['sample_size']))
        .pipe(delta_auc)
    for _ in range(n_samples)]

    lo, hi = bootstrap_ci(estimate, bs_values, ci_level=ci_level)

    return pd.Series(dict(
        estimate=estimate,
        ci_upper=hi,
        ci_lower=lo,
        model=challenger
    ))


@curry
def fast_delta_metric_with_ci(
    data, 
    baseline, 
    challengers, 
    target, 
    *, 
    n_samples=100, 
    ci_level=0.95, 
    weight='weight', 
    metric_fn=roc_auc_score
):
    """
    Similar to fast_delta_metric_with_ci, this function extends fast_metric_with_ci_ to handle multiple predictions. It calculates the metric for each prediction and returns a DataFrame with the results.

    Parameters:
    data (pd.DataFrame): The data containing predictions and targets.
    baseline (str): Column name for baseline predictions.
    challengers (list): List of column names for challenger predictions.
    target (str): Column name for target values.
    n_samples (int): Number of bootstrap samples.
    ci_level (float): Confidence interval level.
    weight (str): Column name for weights.
    metric_fn (callable): Metric function to evaluate.

    Returns:
    pd.DataFrame: Estimated delta metrics and confidence intervals for all challengers.
    """
    fn = fast_delta_metric_with_ci_(
        baseline=baseline,
        n_samples=n_samples,
        ci_level=ci_level,
        target=target,
        weight=weight,
        metric_fn=metric_fn
    )

    all_values = [fn(data=data, challenger=c) for c in challengers]

    return pd.DataFrame(all_values)


@curry
def fast_metric_with_ci(
    data, 
    predictions, 
    target, 
    *, 
    n_samples=100, 
    ci_level=0.95, 
    weight='weight', 
    metric_fn=roc_auc_score
):
    """
    Calculate the metric with confidence intervals for multiple predictions.

    Parameters:
    data (pd.DataFrame): The data containing predictions and targets.
    predictions (list): List of column names for predictions.
    target (str): Column name for target values.
    n_samples (int): Number of bootstrap samples.
    ci_level (float): Confidence interval level.
    weight (str): Column name for weights.
    metric_fn (callable): Metric function to evaluate.

    Returns:
    pd.DataFrame: Estimated metrics and confidence intervals for all predictions.
    """
    fn = fast_metric_with_ci_(
        target=target,
        n_samples=n_samples,
        ci_level=ci_level,
        weight=weight,
        metric_fn=metric_fn
    )

    all_values = [fn(data=data, prediction=p) for p in predictions]

    return pd.DataFrame(all_values)

####################################################################################################################################

#These functions convert between log odds and probabilities. They are used when working with models that output log odds instead of probabilities.

def log_odds_to_proba(x):
    """
    Convert log odds to probability.

    Parameters:
    x (float): Log odds.

    Returns:
    float: Probability.
    """
    return 1 / (1 + np.exp(-x))


def proba_to_log_odds(p):
    """
    Convert probability to log odds.

    Parameters:
    p (float): Probability.

    Returns:
    float: Log odds.
    """
    return np.log(p / (1 - p))

#####################################################################################################################################

def backwards_shap_feature_selection(
    model,
    df_train,
    df_val,
    candidate_features_for_removal,
    target,
    null_hypothesis="feature_is_good",
    fixed_features=[],
    n_features_sample=None,
    extra_validation_sets={},
    sample_weight=None,
    metric_fn=roc_auc_score,
    bootstrap_samples=20,
    ci_level=0.8,
    max_iter=10,
    patience=0,
    max_removals_per_run=None
):
    """
    Perform backwards SHAP feature selection.

    Parameters:
    model (sklearn.base.BaseEstimator): Model to use for feature selection.
    df_train (pd.DataFrame): Training data.
    df_val (pd.DataFrame): Validation data.
    candidate_features_for_removal (list): List of candidate features for removal.
    target (str): Column name for target values.
    null_hypothesis (str): Null hypothesis strategy.
    fixed_features (list): List of fixed features that cannot be removed.
    n_features_sample (int): Number of features to sample in each iteration.
    extra_validation_sets (dict): Extra validation sets for evaluation.
    sample_weight (str): Column name for sample weights.
    metric_fn (callable): Metric function to evaluate.
    bootstrap_samples (int): Number of bootstrap samples.
    ci_level (float): Confidence interval level.
    max_iter (int): Maximum number of iterations.
    patience (int): Patience for stopping criterion.
    max_removals_per_run (int): Maximum number of features to remove per run.

    Returns:
    pd.DataFrame: Logs of the feature selection process.
    """
    valid_nulls = ["feature_is_good", "feature_is_bad"]
    if not null_hypothesis in valid_nulls:
        raise ValueError(f"null_hypothesis should be one of {valid_nulls}, got {null_hypothesis}")

    keys_intersections = set(extra_validation_sets.keys()) & set(candidate_features_for_removal + fixed_features)
    if keys_intersections:
        raise ValueError(f"extra_validation_sets names should not match names of features. Found {keys_intersections}")

    keys_intersections = keys_intersections & set(["metric", "error-contribution"])
    if keys_intersections:
        raise ValueError(f"extra_validation_sets names or feature names should not be 'metric' or 'error-contribution'. Found {keys_intersections}")

    all_logs = []
    p = 0
    for i in tqdm(range(max_iter)):
        all_features = candidate_features_for_removal + fixed_features

        if len(all_features) == 0:
            break

        if (n_features_sample is None) or (len(all_features) <= n_features_sample):
            features_to_use = all_features
        else:
            features_to_use = np.random.choice(all_features, n_features_sample, replace=False)

        run_logs = _backwards_shap_feature_selection(
            model=clone(model),
            df_train=df_train,
            df_val=df_val,
            all_features=features_to_use,
            extra_validation_sets=extra_validation_sets,
            target=target,
            sample_weight=sample_weight,
            metric_fn=metric_fn,
            bootstrap_samples=bootstrap_samples,
            ci_level=ci_level,
        )

        if null_hypothesis == "feature_is_good":
            features_to_remove = (
                run_logs
                [lambda d: d["ci_lower"] > 0]
                [lambda d: d["metric"] == "error-contribution"]
                [lambda d: ~d["model"].isin(fixed_features)]
                .sort_values(by="ci_lower", ascending=False)
            )
        else:
            features_to_remove = (
                run_logs
                [lambda d: d["ci_upper"] > 0]
                [lambda d: d["metric"] == "error-contribution"]
                [lambda d: ~d["model"].isin(fixed_features)]
                .sort_values(by="ci_upper", ascending=False)
            )

        if max_removals_per_run is not None:
            features_to_remove = features_to_remove.iloc[:max_removals_per_run]

        features_to_remove = features_to_remove["model"].values.tolist()  # model means the model without the feature

        run_logs["run_index"] = i
        run_logs["n_features"] = (run_logs["metric"] == "error-contribution").sum()
        run_logs["removed_features"] = str(features_to_remove)
        run_logs["n_features_removed"] = len(features_to_remove)
        all_logs.append(run_logs)

        if len(features_to_remove) == 0:
            if patience:
                if p >= patience:
                    break
                else:
                    p += 1
                    continue
            else:
                break

        candidate_features_for_removal = [i for i in candidate_features_for_removal if not i in features_to_remove]

        p = 0

    if (n_features_sample is not None) and (len(all_features) > n_features_sample):
        run_logs = _backwards_shap_feature_selection(
            model=clone(model),
            df_train=df_train,
            df_val=df_val,
            all_features=all_features,
            extra_validation_sets=extra_validation_sets,
            target=target,
            sample_weight=sample_weight,
            metric_fn=metric_fn,
            bootstrap_samples=bootstrap_samples,
            ci_level=ci_level,
        )
        run_logs["run_index"] = i + 1
        run_logs["n_features"] = len(all_features)
        run_logs["removed_features"] = str([])
        run_logs["n_features_removed"] = 0
        all_logs.append(run_logs)

    return pd.concat(all_logs, ignore_index=True)


def _backwards_shap_feature_selection(
    model,
    df_train,
    df_val,
    all_features,
    extra_validation_sets,
    target,
    sample_weight,
    metric_fn,
    bootstrap_samples,
    ci_level,
):
    """
    Helper function for backwards SHAP feature selection.

    Parameters:
    model (sklearn.base.BaseEstimator): Model to use for feature selection.
    df_train (pd.DataFrame): Training data.
    df_val (pd.DataFrame): Validation data.
    all_features (list): List of all features.
    extra_validation_sets (dict): Extra validation sets for evaluation.
    target (str): Column name for target values.
    sample_weight (str): Column name for sample weights.
    metric_fn (callable): Metric function to evaluate.
    bootstrap_samples (int): Number of bootstrap samples.
    ci_level (float): Confidence interval level.

    Returns:
    pd.DataFrame: Logs of the feature selection process.
    """
    model.fit(df_train[all_features], df_train[target], sample_weight=sample_weight)

    explainer = shap.TreeExplainer(model)
    shap_values_val = explainer.shap_values(df_val[all_features])[-1]

    raw_preds_val = proba_to_log_odds(model.predict_proba(df_val[all_features])[:, -1])

    scores_df = pd.DataFrame(
        log_odds_to_proba(raw_preds_val.reshape(-1, 1) - shap_values_val),
        columns=all_features
    )

    scores_df["val_set"] = raw_preds_val
    scores_df[target] = df_val[target].values
    if sample_weight is not None:
        scores_df["weight__"] = df_val[sample_weight].values

    error_contributions_with_ci = fast_delta_metric_with_ci(
        scores_df,
        baseline="val_set",
        challengers=all_features,
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        target=target,
        weight=sample_weight,
        metric_fn=metric_fn
    ).assign(metric="error-contribution")

    metric = fast_metric_with_ci(
        scores_df,
        predictions=["val_set"],
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        target=target,
        weight=sample_weight,
        metric_fn=metric_fn
    ).assign(metric="metric", used_features=str(all_features))

    extra_val_logs = []
    for k, d in extra_validation_sets.items():
        extra_val_logs.append(
            fast_metric_with_ci(
                d.assign(**{k: lambda d: model.predict_proba(d[all_features])[:, -1], "weight__": lambda d: d[sample_weight] if sample_weight is not None else 1}),
                predictions=[k],
                n_samples=bootstrap_samples,
                ci_level=ci_level,
                target=target,
                weight="weight__",
                metric_fn=metric_fn
            ).assign(metric="metric", used_features=str(all_features))
        )

    return pd.concat([error_contributions_with_ci, metric, *extra_val_logs], ignore_index=True)
