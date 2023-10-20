import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tabulate import tabulate
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

'''
def boxplot_regression(df_, cat_feature_, target_feature_)
- Plots sorted boxplot, how target feature varies with gradations of categorical feature

def get_correlated_attributes(df_, target_feature_, corr_threshold_):
- Selects features, that have correlated coeff "C", such that
|C| > |corr_threshold_|
- Returns a series of correlated attributes

def nan_statistics(df, nan_thresh=0.0)
- Prints out columns and nan percentage.
- Returns dictionary with columns and percentages

def visualize_datasets_distributions(
    dataframes_dict_, 
    columns_, 
    grid_width_=3,
    figwidth_
)
- plots a grid of histograms of grid_width
- column_numbers_ - list of column numbers
- for each column, on the plot there are histograms
for each dataset in dataframes_dict_. To check, that train, validation 
and test data are from same distribution
- Example of usage:
visualize_datasets_distributions(
    {
        'trainval': train_df,
        'test': test_df,
    },
    columns_=list(test_df.columns),
    grid_width_=2,
    figwidth_=10
)

def print_model_cv_scores(sklearn_models_dict_, X_, Y_, cv_, scoring_)
- Uses sklearn cross_val_score() function
- Calculates average cross-validation score and outputs SORTED dictionary of results
- Returns sorted dictionaty with models names and their average CV scores
- Example of usage:
_ = print_model_cv_scores(
    sklearn_models_dict_={
        model_name: model.model for model_name, model in all_models.items()
    },
    X_=X_train_val,
    Y_=Y_train_val,
    cv_=7,
    scoring_='neg_mean_squared_error'
)

def plot_cv_results(sklearn_models_dict_, X_, Y_, cv_, scoring_, to_put_minus_=False)
- Plots cross-validation metrics on seen and unseen data
- Prints the average metrics result on SEEN folds and UNSEEN folds (functionality of print_model_cv_scores() function)
- Fix seed np.random.seed() for reproducible results
- Example of usage:
plot_cv_results(
    sklearn_models_dict_={
        model_name: model.model for model_name, model in all_models.items()
    },
    X_=X_train_val,
    Y_=Y_train_val,
    cv_=5,
    scoring_='neg_mean_squared_error',
    to_put_minus_=True
)

def fit_grid_search(models_dict_, X_, Y_, cv_, scoring_)
- Grid search for all models in the dictionary
- USES MY CLASS MODEL, but not sklearn models!
- Returns dictionary of grid search results
- Example of usage:
grid_search_results = fit_grid_search(
    shortlisted_models,
    X_=X_train_val,
    Y_=Y_train_val,
    cv_ = 5,
    scoring_ = 'neg_mean_squared_error'
)

def fit_randomized_search(models_dict_, X_, Y_, cv_, n_iter_, scoring_)
- Equivalent to fit_grid_search (but with RandomizedSearchCV)

def visualize_regression_predictions(sklearn_models_dict_, X_, Y_, dataset_type_)
- Visualizes predictions of models in the dictionaty, on a given data
- Example of usage:
visualize_regression_predictions(
    models,
    X_=X_train_val,
    Y_=Y_train_val,
    dataset_type_='train set'
)
'''

def boxplot_regression(df_, cat_feature_, target_feature_):
    subset = df_[[cat_feature_, target_feature_]]
    s = subset.groupby([cat_feature_]).median().sort_values(by=target_feature_)
    
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(x=cat_feature_, data=df_, stat='percent')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=cat_feature_, y=target_feature_, data=df_, order=s.index)


def get_correlations(df_, target_feature_, ascending_=False):
    cm = df_.corr()
    return cm[target_feature_].sort_values(ascending=ascending_)


def get_correlated_attributes(df_, target_feature_, corr_threshold_):
    '''
    Selects features, that have correlated coeff "C", such that
    |C| > |corr_threshold_|
    '''
    corrs = get_correlations(df_, target_feature_)
    return corrs.loc[
        (corrs >= abs(corr_threshold_)) |
        (corrs <= -abs(corr_threshold_))
    ]


def nan_percentage(df, colname):
    return (df[colname].isnull().sum() / df.shape[0]) * 100


def nan_statistics(df, nan_thresh=0.0):
    res = {}
    nan_cols = df.loc[:, df.isna().any()].columns
    for col in nan_cols:
        res[col] = nan_percentage(df, col)
    print(f'Col -- Nan percentage')
    for key, val in sorted(res.items(), key=lambda item: item[1], reverse=True):
        if val >= nan_thresh * 100:
            print(key, val)
        else:
            del res[key]
    return res

def nan_report(df, threshold):
    nan_percent_df = df.isna().sum() / df.shape[0]
    subset = nan_percent_df[nan_percent_df > threshold]
    print(f'{subset.shape[0]} / {df.shape[1]} cols ({round(subset.shape[0]/df.shape[1], 2) * 100} %) have nan % > {threshold}\n')
    print(subset)
    return subset

# def inf_statistics(
#     df_,
#     numeric_types_=['float32', 'float64', 'int']
# ):
#     '''
#     Collects only numeric data and checks, if
#     it has infinity values
#     '''
#     numeric_columns = []
#     for numeric_type in numeric_types_:
#         numeric_columns.extend(
#             df_.columns[df_.dtypes == numeric_type].values
#         )
#     res = np.isinf(df_.loc[:, numeric_columns]).sum()
#     print(res)
#     return res

def inf_report(df, inf_threshold):

    inf_percentages = pd.DataFrame(np.isinf(df).sum() / df.shape[0], columns=['inf_percent'])
    inf_percentages = inf_percentages.loc[inf_percentages['inf_percent'] > inf_threshold, :]
    if inf_percentages.size == 0:
        print(f'No infinite values observed')
    else:
        print(f'Following columns have inf percentage > {inf_threshold}')
        print(inf_percentages)
    return inf_percentages

def visualize_datasets_distributions(
    dataframes_dict,
    columns,
    grid_width=3,
    figwidth=10
):
    print(f'Visualizing datasets distributions')
    n_plots = len(columns)
    if n_plots % grid_width == 0:
        grid_height = int(n_plots / grid_width)
    else:
        grid_height = int(n_plots / grid_width) + 1
        
    
    HEIGHT_RESOLUTION = 3.2

    _, ax = plt.subplots(
        nrows=grid_height,
        ncols=grid_width,
        figsize=(figwidth, int(HEIGHT_RESOLUTION * grid_height))
    )
    if grid_height == grid_width == 1:
        ax = np.array([[ax]])
    elif grid_width == 1:
        ax = np.expand_dims(ax, axis=1)
    elif grid_height == 1:
        ax = np.expand_dims(ax, axis=0)


    print(ax.shape, type(ax))
    for i in range(grid_height):
        for j in range(grid_width):
            cur_column_number = i * (grid_width) + j
            
            if cur_column_number >= n_plots:
                return

            columns_data = {}
            for dataset_name, df in dataframes_dict.items():

                columns_data[dataset_name] = \
                    df.loc[:, columns[cur_column_number]].values

            for dataset_name, data in columns_data.items():
                ax[i, j].hist(data, density=True, alpha=0.3, label=dataset_name)

            ax[i, j].set_title(f'{columns[cur_column_number]}')
            ax[i, j].legend()

def print_model_cv_scores(sklearn_models_dict_, X_, Y_, cv_, scoring_):
    res = {}
    for name, model in sklearn_models_dict_.items():
        scores = cross_val_score(
            model,
            X_,
            Y_,
            cv=cv_,
            scoring=scoring_
        )
        res[name] = scores
    
    # Sort the dict
    sorted_res = {
        k:v for \
        k, v in sorted(res.items(), key = lambda item: np.mean(item[1]))
    }
    for model_name, scores in sorted_res.items():
        print(f'Model: {model_name}, mean: {np.mean(scores)}, std: {np.std(scores)}')

    return sorted_res

def visualize_target_vs_columns(df, target_colname, columns, grid_width=5, width_scale=3.0, height_scale=3.0):
    grid_height = len(columns) // grid_width if len(columns) % grid_width == 0 \
        else len(columns) // grid_width + 1

    # create the figure
    fig, axes = plt.subplots(ncols=grid_width, nrows=grid_height, figsize=(
        int(width_scale * grid_width), int(height_scale * grid_height))
    )

    # flatten the axes to make it easier to index
    axes = axes.flatten()

    # iterate through the column values, and use i to index the axes
    for i, v in enumerate(columns):
        
        # plot the actual price against the features
        axes[i].scatter(x=df[v], y=df[target_colname], s=35, ec='white', label='actual')
        
        # set the title and ylabel
        axes[i].set(title=f'Feature: {v}', ylabel='price')
    plt.tight_layout()

def _print_sorted_results(cv_metrics_results_):
    metrics_dict = {
        model_name: {
            'test_score': result['test_score'],
            'train_score': result['train_score']
        }
        for model_name, result in cv_metrics_results_.items()
    }

    # Sort based on test score
    metrics_dict_sorted = {
        k: v for
        k, v in sorted(
            metrics_dict.items(),
            key=lambda item: np.mean(item[1]['test_score'])
        )
    }

    metrics_averaged_sorted = {
        model_name: (np.mean(result['train_score']),
                     np.std(result['train_score']),
                     np.mean(result['test_score']),
                     np.std(result['test_score']))
        for model_name, result in metrics_dict_sorted.items()
    }
    headers = [
        'Model',
        'Seen folds avg score',
        'Seen folds std',
        'Unseen folds avg score',
        'Unseen folds std'
    ]
    print(
        tabulate(
            [(k,) + v for k, v in metrics_averaged_sorted.items()],
            headers=headers
        )
    )


def plot_cv_results(
    sklearn_models_dict_,
    X_,
    Y_,
    cv_,
    scoring_,
    to_put_minus_=False
):

    cv_metrics_results = {}

    for model_name, model in sklearn_models_dict_.items():
        cv_res = cross_validate(
            model,
            X_,
            Y_,
            cv=cv_,
            scoring=scoring_,
            return_train_score=True
        )
        _, ax = plt.subplots()
        x = np.arange(len(cv_res['test_score']))
        width = 0.5
        if to_put_minus_:
            train_score = -cv_res['train_score']
            test_score = -cv_res['test_score']
        else:
            train_score = cv_res['train_score']
            test_score = cv_res['test_score']

        cv_metrics_results[model_name] = cv_res

        ax.bar(x - width / 2, test_score, width, label='validation')
        ax.bar(x + width / 2, train_score, width, label='train')

        ax.set_title(f'Results for {model_name}')
        ax.set_xlabel(f'CV fold number')
        ax.set_ylabel(f'Metrics: {scoring_}')

        ax.legend()
        ax.grid()

    _print_sorted_results(cv_metrics_results)

    return cv_metrics_results


def _collect_train_test_scores(cv_results_, best_estimator_index_):

    n_splits = len(
        [k for k in list(cv_results_.keys())
            if ('train_score' in k) and ('split') in k])
    train_scores = []
    test_scores = []

    for split in np.arange(n_splits):
        train_split_key = f'split{split}_train_score'
        test_split_key = f'split{split}_test_score'

        best_estimator_train_score = \
            cv_results_[train_split_key][best_estimator_index_]
        best_estimator_test_score = \
            cv_results_[test_split_key][best_estimator_index_]

        train_scores.append(best_estimator_train_score)
        test_scores.append(best_estimator_test_score)
    return train_scores, test_scores


def fit_grid_search(
    models_dict_,
    X_,
    Y_,
    cv_,
    scoring_,
    **kwargs
):
    res = {
        name: None for name in list(models_dict_.keys())
    }
    for name, model in models_dict_.items():
        print(f'Fitting {name}')

        grid_search_estimator = GridSearchCV(
            model.model,
            param_grid=model.grid_search_param_grid,
            cv=cv_,
            scoring=scoring_,
            return_train_score=True,
            refit=True,
            verbose=kwargs.get('verbose') or 0
        )

        grid_search_result = grid_search_estimator.fit(X_, Y_)
        res[name] = grid_search_result

        train_scores, test_scores = _collect_train_test_scores(
            grid_search_result.cv_results_,
            grid_search_result.best_index_)
        assert len(train_scores) == len(test_scores)

        # Plotting
        x = np.arange(1, len(train_scores) + 1)
        WIDTH = 0.5
        _, ax = plt.subplots()
        ax.bar(x - WIDTH / 2, test_scores, WIDTH, label='validation')
        ax.bar(x + WIDTH / 2, train_scores, WIDTH, label='train')
        ax.set_title(f'{name} (best_estimator) cross validation')
        ax.legend(loc='lower right')
        ax.set_xlabel('Number of fold')
        ax.set_ylabel('Metrics')
        ax.grid()

    return res


def fit_randomized_search(
    models_dict_,
    X_,
    Y_,
    cv_,
    n_iter_,
    scoring_,
    **kwargs
):
    RANDOM_STATE = 42
    res = {}
    for name, model in models_dict_.items():
        print(f'Fitting {name}')

        estimator = RandomizedSearchCV(
            model.model,
            param_distributions=model.random_search_param_grid,
            cv=cv_,
            n_iter=n_iter_,
            scoring=scoring_,
            return_train_score=True,
            refit=True,
            random_state=RANDOM_STATE,
            verbose=kwargs.get('verbose') or 0
        )

        rand_search_res = estimator.fit(X_, Y_)
        res[name] = rand_search_res
    return res


# def visualize_regression_predictions(
#     sklearn_models_dict_,
#     X_,
#     Y_,
#     dataset_type_
# ):
#     _, ax = plt.subplots()
#     ax.plot(
#         Y_,
#         label=f'{dataset_type_} target'
#     )

#     for model_name, model in sklearn_models_dict_.items():
#         predictions = model.predict(X_)
#         ax.scatter(
#             x=np.arange(len(predictions)),
#             y=predictions,
#             label=f'{model_name} predictions'
#         )

#     ax.set_xlabel('Dataset instance')
#     ax.set_ylabel('Prediction')
#     ax.set_title(f'Visualized predictons on {dataset_type_}')

#     ax.legend()
#     ax.grid()

def visualize_predictions(model, X, Y):
    fig, ax = plt.subplots()
    predictions=model.predict(X)
    ax.scatter(x=predictions, y=Y)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')
    
    max_val = max(np.max(predictions), np.max(Y))
    min_val = min(np.min(predictions), np.min(Y))

    ax.plot(np.linspace(min_val, max_val, 50), np.linspace(min_val, max_val, 50), color='red', label='x==y')
    ax.grid()
    ax.legend()
    

def dict_subset(dict_, keys_):
    available_keys = dict_.keys()
    res = {}
    for key in keys_:
        if key in available_keys:
            res.update(
                {
                    key: dict_[key]
                }
            )
        else:
            print(f'Did not find {key} in dictionary')
    return res

def report_feature_histograms(df, n_cols, columns, figsize=(20, 50), hist_params={}):
    n_features = df.shape[1]
    temp = n_features // n_cols
    n_rows = temp if n_features % n_cols == 0 else temp + 1

    _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    ax = ax.flatten()

    for i, col in enumerate(columns):
        _ = ax[i].hist(df[col], **hist_params)
        ax[i].set_title(col)

def bin_column(df, column_name, bins, to_plot=True):
    # For feature binning (bucketizing)
    labels = np.arange(len(bins) - 1)

    _, ax = plt.subplots()
    ax.hist(df[column_name])
    for bin in bins:
        ax.axvline(bin, linestyle='--', color='red')
    ax.set_title(column_name)

    df[f'{column_name}_binned'] = pd.cut(
        df[column_name],
        bins=bins,
        labels=labels
    )

def do_feature_cross(df, column_name, trained_one_hot_encoder):
    assert column_name in df.columns, print(f'{column_name} is not in columns')

    categories_in_df = df[column_name].unique()
    for cat in categories_in_df:
        assert cat in trained_one_hot_encoder.categories_[0], \
            print(f'Category {cat} is not learned, but is in the dataframe')
    
    print(f'NANs before preprocessing: {df.isna().sum().sum()}')
    print(f'Shape before: {df.shape}')
    transformed_df = pd.DataFrame(encoder.transform(df[[column_name]]).toarray())
    
    # Aligning index to avoid NAN after join
    transformed_df.index = df.index
    df = df.join(transformed_df)
    print(f'NANs after preprocessing: {df.isna().sum().sum()}')
    print(f'Shape after: {df.shape}')
    return df

def plot_confusion_matrix(Y_true, Y_pred, labels, figsize=(15, 8)):
    conf_mat = confusion_matrix(y_true=Y_true, y_pred=Y_pred)
    print(f'Rows: groundtruth classes, columns: predicted classes!')

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.flatten()
    ax[0].set_title(f'Confusion mat. \n with no highlight by A.Geron')

    c = ConfusionMatrixDisplay(conf_mat, display_labels=labels)
    
    c.plot(ax=ax[0])
    ax[0].set_xticklabels(labels, rotation=90)
    

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mat / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    
    ConfusionMatrixDisplay(norm_conf_mx, display_labels=labels).plot(ax=ax[1])
    ax[1].set_title(f'Confusion mat. \n with highlight by A.Geron')
    ax[1].set_xticklabels(labels, rotation=90)

    
    fig.delaxes(fig.axes[3])
    fig.delaxes(fig.axes[2])


def output_classification_mistakes(Y_predicted, Y_true, label_encoder):
    idxs = np.where(Y_predicted != Y_true)[0]
    for idx in idxs:
        # print(idx, predictions[idx])
        print(f'''
        Predicted: {label_encoder.inverse_transform(Y_predicted[[idx]])},
        Actual: {label_encoder.inverse_transform(Y_true[[idx]])}
        ''')

def plot_correlation_matrix(df, delete_diagonals=True):
    corr_mx = df.corr()
    if delete_diagonals:
        n = corr_mx.shape[0]
        corr_mx.values[range(n), range(n)] = 0
    sns.heatmap(corr_mx)
    return corr_mx
