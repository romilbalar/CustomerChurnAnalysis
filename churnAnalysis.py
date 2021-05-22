
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.feature_selection import SelectFromModel


df = pd.read_csv('Churn_Modelling.csv')


df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
new_names = {
    'CreditScore': 'credit_score',
    'Geography': 'country',
    'Gender': 'gender',
    'Age': 'age',
    'Tenure': 'tenure',
    'Balance': 'balance',
    'NumOfProducts': 'number_products',
    'HasCrCard': 'owns_credit_card',
    'IsActiveMember': 'is_active_member',
    'EstimatedSalary': 'estimated_salary',
    'Exited': 'exited'
}
df.rename(columns=new_names, inplace=True)
df.head()


#exploratory data anaylysis

amount_retained = df[df['exited'] == 0]['exited'].count() / df.shape[0] * 100
amount_lost = df[df['exited'] == 1]['exited'].count() / df.shape[0] * 100
fig, ax = plt.subplots()
sns.countplot(x='exited', palette="Set3", data=df)
plt.xticks([0, 1], ['Retained', 'Lost'])
plt.xlabel('Condition', size=15, labelpad=12, color='grey')
plt.ylabel('Amount of customers', size=15, labelpad=12, color='grey')
plt.title("Proportion of customers lost and retained", size=15, pad=20)
plt.ylim(0, 9000)
plt.text(-0.15, 7000, f"{round(amount_retained, 2)}%", fontsize=12)
plt.text(0.85, 1000, f"{round(amount_lost, 2)}%", fontsize=12)
sns.despine()
plt.show()

categorical_labels = [['gender', 'country'], ['owns_credit_card', 'is_active_member']]
colors = [['Set1', 'Set2'], ['Set3', 'PuRd']]

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for i in range(2):
    for j in range(2):
        feature = categorical_labels[i][j]
        color = colors[i][j]
        ax1 = sns.countplot(x=feature, hue='exited', palette=color, data=df, ax=ax[i][j])
        ax1.set_xlabel(feature, labelpad=10)
        ax1.set_ylim(0, 6000)
        ax1.legend(title='Exited', labels= ['No', 'Yes'])
        if i == 1:
            ax1.set_xticklabels(['No', 'Yes'])
sns.despine()



female_churn = round(df[(df['exited'] == 1) & (df['gender'] == 'Female')]['exited'].count() / df[df['gender'] == 'Female']['exited'].count()*100, 2)
male_churn = round(df[(df['exited'] == 1) & (df['gender'] == 'Male')]['exited'].count() / df[df['gender'] == 'Male']['exited'].count() * 100, 2)

print(f"The percentage of female customers churning is {female_churn}% while the percetage of male customers churning is {male_churn}%")


active_churn = round(df[(df['exited'] == 1) & (df['is_active_member'] == 1)]['exited'].count() / df[df['is_active_member'] == 1]['exited'].count()*100, 2)
inactive_churn = round(df[(df['exited'] == 1) & (df['is_active_member'] == 0)]['exited'].count() / df[df['is_active_member'] == 0]['exited'].count() * 100, 2)

print(f"The percentage of active members churning is {active_churn}% while the percetage of inactive members churning is {inactive_churn}%")

credit_churn = round(df[(df['exited'] == 1) & (df['owns_credit_card'] == 1)]['exited'].count() / df[df['owns_credit_card'] == 1]['exited'].count()*100, 2)
no_credit_churn = round(df[(df['exited'] == 1) & (df['owns_credit_card'] == 0)]['exited'].count() / df[df['owns_credit_card'] == 0]['exited'].count() * 100, 2)

print(f"The percentage of custumers with credit card churning is {credit_churn}% while the percetage that do not have credit cards and")
print(f"churn is {no_credit_churn}%")


numerical_labels = [['age', 'credit_score'], 
                    ['tenure', 'balance'],
                   ['number_products', 'estimated_salary']]
num_colors = [['Set1', 'Set2'], 
              ['Set3', 'PuRd'],
              ['Spectral', 'Wistia']]

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
for i in range(3):
    for j in range(2):
        feature = numerical_labels[i][j]
        color = num_colors[i][j]
        ax1 = sns.boxplot(x='exited', y=feature, palette=color, data=df, ax=ax[i][j])
        ax1.set_xlabel('Exited', labelpad=10)
        ax1.set_xticklabels(['No', 'Yes'])
sns.despine()

pd.DataFrame(df.groupby('exited')['age'].describe())
pd.DataFrame(df.groupby('exited')['balance'].describe())
sns.pairplot(df, vars=['age', 'credit_score', 'balance', 'estimated_salary'], 
             hue="exited", palette='husl')
sns.despine()


#feature creation


df['creditscore_age_ratio'] = df['credit_score'] / df['age']
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(y='creditscore_age_ratio', x='exited', palette='summer', data=df)
ax.set_xticklabels(['No', 'Yes'])
sns.despine()
df['balance_salary_ratio'] = df['balance'] / df['estimated_salary']
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(y='balance_salary_ratio', x='exited', palette='winter', data=df)
ax.set_xticklabels(['No', 'Yes'])
ax.set_ylim(-1, 6)
sns.despine()


x_drop = ['exited', 'estimated_salary', 'balance', 'age', 'credit_score']

x = df.drop(x_drop, axis=1)
y = df['exited']

for label in ['gender', 'country']:
    le = LabelEncoder()
    le.fit(x[label])
    print(le.classes_)
    x.loc[:, label] = le.transform(x[label])

x['gender'].unique()
x['country'].unique()
features = x.columns
x = np.array(x)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, 
                                                    shuffle=True, stratify=y)



#model fitting
def print_best_model(model):
    print(f"The best parameters are: {model.best_params_}")
    print(f"The best model score is: {model.best_score_}")    
    print(f"The best estimator is: {model.best_estimator_}")


def get_scores(y, predicted, predicted_proba):
    auc_score = roc_auc_score(y, predicted)
    fpr_df, tpr_df, _ = roc_curve(y, predicted_proba) 
    return auc_score, fpr_df, tpr_df

def get_confusion_matrix(y_test, y_predicted):
    plt.figure()
    random_confusion = confusion_matrix(y_test, y_predicted)
    ax = sns.heatmap(random_confusion, annot=True, cmap="YlGnBu");
    ax.set_ylim([0,2]);


#logistic regression

param_grid_log = {
    'C': [0.1, 1, 10, 50, 100, 200],
    'max_iter': [200, 300],
    'penalty': ['l2'],
    'tol':[0.00001, 0.0001],
}

log_first = LogisticRegression(solver='lbfgs')

log_grid = GridSearchCV(log_first, param_grid=param_grid_log, cv=10, verbose=1, n_jobs=-2)

log_grid.fit(x, y)

print_best_model(log_grid)

param_grid_svm = {
    'C': [0.5, 100, 150],
    'kernel': ['rbf'],
    'gamma': [0.1, 0.01, 0.001],
    'probability': [True]
}

svm_first = SVC()

svm_grid = GridSearchCV(svm_first, param_grid=param_grid_svm, cv=3, verbose=3, n_jobs=-2)

svm_grid.fit(x, y)

print_best_model(svm_grid)

param_grid = {'max_depth': [3, 5, 6], 
              'max_features': [2, 4, 6],
              'n_estimators':[50, 100],
              'min_samples_split': [3, 5, 7]}

random_forest = RandomForestClassifier()

random_forest_grid = GridSearchCV(random_forest, param_grid, cv=5, refit=True, verbose=3, n_jobs=-2)

random_forest_grid.fit(x, y)

print_best_model(random_forest_grid)

best_log_estimator = LogisticRegression(C=100, max_iter=200, penalty='l2', tol=1e-05, solver='lbfgs')

best_log_estimator.fit(X_train, y_train)

best_svm_estimator = SVC(C=100, gamma=0.01, kernel='rbf', probability=True)

best_svm_estimator.fit(X_train, y_train)

best_rf_estimator = RandomForestClassifier(max_depth=6, max_features=6, min_samples_split=3, n_estimators=100)

best_rf_estimator.fit(X_train, y_train)

log_predict_train = best_log_estimator.predict(X_train)

log_predict_test = best_log_estimator.predict(X_test)

accuracy_score(y_train, log_predict_train)

print(classification_report(y_train, log_predict_train))


accuracy_score(y_test, log_predict_test)
print(classification_report(y_test, log_predict_test))
get_confusion_matrix(y_test, log_predict_test)

svm_predict_train = best_svm_estimator.predict(X_train)

svm_predict_test = best_svm_estimator.predict(X_test)

accuracy_score(y_train, svm_predict_train)
print(classification_report(y_train, svm_predict_train))

accuracy_score(y_test, svm_predict_test)

print(classification_report(y_test, svm_predict_test))
get_confusion_matrix(y_test, svm_predict_test)
rf_predict_train = best_rf_estimator.predict(X_train)

rf_predict_test = best_rf_estimator.predict(X_test)
accuracy_score(y_train, rf_predict_train)
print(classification_report(y_train, rf_predict_train))
accuracy_score(y_test, rf_predict_test)
print(classification_report(y_test, rf_predict_test))
get_confusion_matrix(y_test, rf_predict_test)
auc_log, fpr_log, tpr_log = get_scores(y, best_log_estimator.predict(x), best_log_estimator.predict_proba(x)[:,1])
auc_svm, fpr_svm, tpr_svm = get_scores(y, best_svm_estimator.predict(x), best_svm_estimator.predict_proba(x)[:,1])
auc_rf, fpr_rf, tpr_rf = get_scores(y, best_rf_estimator.predict(x), best_rf_estimator.predict_proba(x)[:,1])
plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_log, tpr_log, label = f'Logistic Regression Score: {str(round(auc_log, 3))}', color='#FA8072')
plt.plot(fpr_svm, tpr_svm, label = f'SVM RBF Score: {str(round(auc_svm, 3))}', color='#82E0AA')
plt.plot(fpr_rf, tpr_rf, label = f'Random Forest Score: {str(round(auc_rf, 3))}', color='#A569BD')
plt.plot([0,1], [0,1], '--', label = 'score 0.5', color='#34495E')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()



#feature importance

def get_most_important_features(model, num_features, model_type):
    selector = SelectFromModel(model, threshold=-np.inf, max_features=num_features)
    selector.fit(X_train, y_train)
    if model_type == "logistic":
        features_idx = selector.get_support()
        features_name = features[features_idx]
    else:
        return "It is not possible to get attributes"
    return features_name

get_most_important_features(model=best_log_estimator, num_features=4, model_type="logistic")


def feature_importance(model, feature_list):
    """
    Function that gets and plots the feature importance
    for the given model
    :param model: the model to evaluaate
    :param feature_list: a list of features contained in the model

    :returns a plot with feature importance
    """
    #Get the list of feaature importance from the model
    importances = list(model.feature_importances_)
    #zip together feature names and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    #sort the feature importance by importance
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    #Print the list of feature importance
    [print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances];
    #set colors for the plot
    colors = cm.rainbow(np.linspace(0, 1, len(feature_list)))
    
    #get the list of features sorted
    characteristics = [x[0] for x in feature_importances]
    #get the list of importance sorted
    importances_plot = [x[1] for x in feature_importances]
    #plot in a bar plot
    plt.bar(characteristics, importances_plot, color=colors)
    #adjust characteristics of the plot
    plt.xticks(list(range(len(characteristics))), characteristics, rotation = 90)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gcf().subplots_adjust(bottom=0.3);

feature_importance(best_rf_estimator, features)
