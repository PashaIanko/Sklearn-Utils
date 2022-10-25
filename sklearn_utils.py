import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tabulate import tabulate

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


def visualize_datasets_distributions(
    dataframes_dict_,
    columns_,
    grid_width_=3,
    figwidth_=10
):
    print(f'Visualizing datasets distributions')
    n_plots = len(columns_)
    if n_plots % grid_width_ == 0:
        grid_height = int(n_plots / grid_width_)
    else:
        grid_height = int(n_plots / grid_width_) + 1
        
    
    HEIGHT_RESOLUTION = 3.2

    _, ax = plt.subplots(
        grid_height,
        grid_width_,
        figsize=(figwidth_, int(HEIGHT_RESOLUTION * grid_height))
    )

    for i in range(grid_height):
        for j in range(grid_width_):
            cur_column_number = i * (grid_width_) + j
            
            if cur_column_number >= n_plots:
                return

            columns_data = {}
            for dataset_name, df in dataframes_dict_.items():

                columns_data[dataset_name] = \
                    df.loc[:, columns_[cur_column_number]].values

            for dataset_name, data in columns_data.items():
                ax[i, j].hist(data, density=True, alpha=0.3, label=dataset_name)

            ax[i, j].set_title(f'{columns_[cur_column_number]}')
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


def fit_grid_search(models_dict_, X_, Y_, cv_, scoring_):
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
            refit=True
        )

        grid_search_result = grid_search_estimator.fit(X_, Y_)
        res[name] = grid_search_result
    return res


def fit_randomized_search(models_dict_, X_, Y_, cv_, n_iter_, scoring_):
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
            random_state=RANDOM_STATE
        )

        rand_search_res = estimator.fit(X_, Y_)
        res[name] = rand_search_res
    return res


def visualize_regression_predictions(
    sklearn_models_dict_,
    X_,
    Y_,
    dataset_type_
):
    _, ax = plt.subplots()
    ax.plot(
        Y_,
        label=f'{dataset_type_} target'
    )

    for model_name, model in sklearn_models_dict_.items():
        predictions = model.predict(X_)
        ax.scatter(
            x=np.arange(len(predictions)),
            y=predictions,
            label=f'{model_name} predictions'
        )

    ax.set_xlabel('Dataset instance')
    ax.set_ylabel('Prediction')
    ax.set_title(f'Visualized predictons on {dataset_type_}')

    ax.legend()
    ax.grid()
    

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
