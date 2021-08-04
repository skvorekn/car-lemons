# car_lemons

Case study comparing prediction methods with re-usable template code

Example research question: predict if car purchased at an auction is a kick to provide best inventory selection to their customers from [Kaggle](https://www.kaggle.com/c/DontGetKicked)

# Models

* Random Forest tuned using random search
    - Intuition: RF is a good choice for data with a high number of observations vs. features. It is a low bias, high variance method. It has good interpretability vs. SVM, and potentially better performing than a Lasso model due to its nonlinear nature.
* KNN
* Kernel SVM
* Light Gradient Boosted Machine
* Lasso

# Development

## Data

In the data/ folder, run:
```
kaggle competitions download -c DontGetKicked
```

or download manually.

## Environment

* pipenv

## Testing

* pytest

# Contributing

CI: formatting, pytests pass & coverage, HISTORY updated