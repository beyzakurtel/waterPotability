from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

# Define plot_confusion_matrix function
def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix - {model_name}')
    st.pyplot(fig)

# Define plot_learning_curve function
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    st.pyplot(plt)

st.title("Water Potability Prediction")

# Load data
df = pd.read_csv("data/water_potability.csv")
st.write("Water Potability Data Set:")
st.write(df)

# Show missing values before imputation
missing_col = df.isnull().sum()
fig = go.Figure(data=[go.Pie(
    labels=missing_col.index,
    values=missing_col.values,
    hole=0.4,
    marker=dict(colors=['lightpink', 'lightsalmon', 'lightgreen', 'lightcoral', 'lightblue', 'lightred', 'darkpurple', 'lightpurple']),
)])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Distribution of missing data', template='plotly_white')
st.plotly_chart(fig)



# Function to impute mean values
def impute_mean(row):
    for column in df.columns:
        if pd.isna(row[column]) and column != 'Potability':
            row[column] = means.loc[row['Potability'], column]
    return row

means = df.groupby('Potability').mean()
df = df.apply(impute_mean, axis=1)





# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)


st.write("Filling in missing data based on averages:")
st.write(df)



# Plot histograms for each feature
st.write("Feature Distributions:")
plt.figure(figsize=(12, 12))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(x=df[col], hue=df["Potability"], multiple="dodge")
    plt.title(f"{col} Verilerinin Dağılımı")
    plt.tight_layout()
    plt.xticks(rotation=90)
st.pyplot(plt)


st.write("Normalized Data:")
st.write(normalized_df)

# Feature selection using RFE
X_train, X_test, y_train, y_test = train_test_split(normalized_df.drop('Potability', axis=1), normalized_df['Potability'], test_size=0.2, random_state=42)
rfe_model = RandomForestClassifier()
rfe = RFE(estimator=rfe_model, n_features_to_select=5)
rfe.fit(X_train, y_train)

selected_features_indexes_rfe = rfe.get_support(indices=True)
selected_features_names_rfe = normalized_df.drop('Potability', axis=1).columns[selected_features_indexes_rfe]
X_train_rfe = X_train.iloc[:, selected_features_indexes_rfe]
X_test_rfe = X_test.iloc[:, selected_features_indexes_rfe]

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50],
    'max_depth': [5],
    'min_samples_leaf': [5]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_rfe, y_train)
best_rf_model = grid_search.best_estimator_

# Train the best Random Forest model
best_rf_model.fit(X_train_rfe, y_train)
rf_pred = best_rf_model.predict(X_test_rfe)
rf_cm = confusion_matrix(y_test, rf_pred)
rf_acc = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

rf_scores = cross_val_score(best_rf_model, X_train_rfe, y_train, cv=5, scoring='accuracy')
rf_mean_accuracy = np.mean(rf_scores)



st.title("Random Forest with RFE")
st.write()

# Display confusion matrix
st.write("Confusion Matrix - Random Forest with RFE (Top 5 Features)")
plot_confusion_matrix(rf_cm, "Random Forest with RFE (Top 5 Features)")

# Display metrics
st.write("Random Forest Metrics:")
st.write(f"Accuracy: {rf_acc}")
st.write(f"Precision: {rf_precision}")
st.write(f"Recall: {rf_recall}")
st.write(f"F1 Score: {rf_f1}")

# Display mean accuracy with 5-fold cross-validation
st.write(f"Random Forest Mean Accuracy (5-fold CV): {rf_mean_accuracy}")

# Plot learning curve
st.write("Learning Curve:")
plot_learning_curve(best_rf_model, "Learning Curve - Random Forest with RFE", X_train_rfe, y_train, cv=5)
