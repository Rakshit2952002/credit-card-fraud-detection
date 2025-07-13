
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Assumes train_df, test_df, predictors, and target are already defined

# Define the models
rf_model = RandomForestClassifier(n_jobs=4, random_state=2018)
lr_model = LogisticRegression()

# Train the models
rf_model.fit(train_df[predictors], train_df[target])
lr_model.fit(train_df[predictors], train_df[target])

# Get predicted probabilities
rf_probs = rf_model.predict_proba(test_df[predictors])
lr_probs = lr_model.predict_proba(test_df[predictors])

# ROC AUC scores
rf_roc_auc = roc_auc_score(test_df[target], rf_probs[:, 1])
print("Random Forest ROC AUC:", rf_roc_auc)

lr_roc_auc = roc_auc_score(test_df[target], lr_probs[:, 1])
print("Logistic Regression ROC AUC:", lr_roc_auc)

# Combined prediction
combined_probs = (rf_probs[:, 1] + lr_probs[:, 1]) / 2
combined_roc_auc = roc_auc_score(test_df[target], combined_probs)
print("Hybrid Model ROC AUC:", combined_roc_auc)
