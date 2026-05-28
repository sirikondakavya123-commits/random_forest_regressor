# IMPORT LIBRARIES

import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# LOAD CLEANED DATASET

df = pd.read_csv(
    "../data/cleaned_housing.csv"
)

print("Dataset Loaded Successfully")

# FEATURES AND TARGET

X = df.drop(
    "median_house_value",
    axis=1
)

y = df["median_house_value"]

print("Features and Target Selected")

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

# MODEL

model = RandomForestRegressor(
    random_state=42
)

# HYPERPARAMETER TUNING

params = {

    "n_estimators": [50, 100],

    "max_depth": [5, 10, 15],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2]

}

grid_search = GridSearchCV(

    estimator=model,

    param_grid=params,

    cv=5,

    scoring="r2",

    n_jobs=-1,

    verbose=2

)

grid_search.fit(
    X_train,
    y_train
)

# BEST MODEL

best_model = grid_search.best_estimator_

print("Best Parameters")

print(grid_search.best_params_)

# PREDICTION

y_pred = best_model.predict(
    X_test
)

# EVALUATION

print("MAE")

print(
    mean_absolute_error(
        y_test,
        y_pred
    )
)

print("MSE")

print(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("R2 SCORE")

print(
    r2_score(
        y_test,
        y_pred
    )
)

# SAVE MODEL

pickle.dump(

    best_model,

    open(
        "../models/random_forest_regressor.pkl",
        "wb"
    )

)

print("Model Saved Successfully")