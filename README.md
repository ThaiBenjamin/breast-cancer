# Breast Cancer Classifier

A workshop exercise in supervised learning: training a logistic regression classifier on
the Wisconsin Breast Cancer Dataset to predict whether a tumor is malignant or benign.

The dataset has 30 numeric features — cell radius, texture, perimeter, and so on — and a
binary target, which makes it a clean setup for working through classification end to end.

## What the script does

Loads the CSV with pandas, drops the columns that aren't predictive, and encodes the
categorical label. Splits into train and test sets with `train_test_split` so evaluation
happens on data the model hasn't seen. Wraps `LogisticRegression` in a pipeline behind a
`StandardScaler`, fits it, and reports accuracy along with precision, recall, and F1 via
`classification_report`. The trained model is wrapped in a small function that takes a
single sample and returns a prediction.

## What I took from it

Using a real medical dataset made this land harder than a toy example would have.

The lesson that stuck was why feature scaling matters. Logistic regression trained by
gradient descent converges much faster and more reliably when all the features are on the
same scale, and with 30 features measured in completely different units, that's not
optional. Bundling `StandardScaler` into the pipeline rather than scaling by hand also means
the test data gets the transform fit on the training data, instead of leaking.

`classification_report` taught me the other half: accuracy alone is misleading here. On
medical data you care much more about false negatives than false positives, and a single
accuracy number hides that distinction entirely.
