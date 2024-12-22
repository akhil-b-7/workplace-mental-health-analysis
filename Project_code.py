#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')


df = pd.read_csv('survey.csv')
df.head()

print('Shape of the data: ')
df.shape

print('To check out for null values')
df.isna().sum()

print('Dropping uncessary columns')
df.drop(['Country','state','comments'], axis=1, inplace=True)

df.dropna(how='any', axis=0, inplace=True)

df.head()

df.info()

for col in df.columns:
    print('Unique values in {} :'.format(col),len(df[col].unique()))

print('Checking for irrelevancy: ')
df['Age'].unique()

df['Age'].replace([df['Age'][df['Age'] < 18]], np.nan, inplace = True)
df['Age'].replace([df['Age'][df['Age'] > 72]], np.nan, inplace = True)

df['Age'].median()

df['Gender'].unique()

df['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male','Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                    'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

df['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female','femail', 'Cis Female', 'cis-female/femme', 
                    'Femake', 'Female (cis)','woman',], 'Female', inplace = True)

df["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary','fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All','ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?','Guy (-ish) ^_^', 'Trans woman',], 'Others', inplace = True)

px.histogram(df, x='Age', color='Gender')

# Treatment distribution by Age and Gender
import plotly.express as px

fig = px.histogram(df, x='Age', color='treatment', facet_col='Gender',
                   title="Treatment Distribution by Age and Gender",
                   labels={'treatment': 'Received Treatment'})
fig.show()

# Work interference vs treatment
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='work_interfere', hue='treatment', palette='Set2')
plt.title("Work Interference vs Treatment")
plt.xlabel("Work Interference")
plt.ylabel("Count")
plt.legend(title="Treatment", loc="upper right")
plt.show()

# Mental health consequence by gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='mental_health_consequence', hue='Gender', palette='coolwarm')
plt.title("Mental Health Consequences by Gender")
plt.xlabel("Mental Health Consequences")
plt.ylabel("Count")
plt.legend(title="Gender", loc="upper right")
plt.show()

# # Data Summary
df.describe

df.columns

# Treatment by Age

for i in ['Male', 'Female', 'Others']:
    t = df[df['Gender']==i].copy()
    fig = px.histogram(t, x='Age',nbins=40,color='treatment')
    fig.update_layout(
    title=i)
    fig.show()

px.histogram(df, x='Age',nbins=40)

# Wellness Program by Age

for i in ['Male', 'Female', 'Others']:
    t = df[df['Gender']==i].copy()
    fig = px.histogram(t, x='Age',nbins=40,color='wellness_program')
    fig.update_layout(
    title=i)
    fig.show()

# # ML Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import binarize, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score

data = pd.read_csv('survey.csv')

defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

for feature in data:
    if feature in intFeatures:
        data[feature] = data[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        data[feature] = data[feature].fillna(defaultString)
    elif feature in floatFeatures:
        data[feature] = data[feature].fillna(defaultFloat)
    else:
        print('Error : Feature %s not recognized.' % feature)

data.head()

gender = data['Gender'].str.lower()

gender = data['Gender'].unique()

male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]

trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]

female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in data.iterrows():
    if str.lower(col.Gender) in male_str:
        data['Gender'].replace(to_replace = col.Gender, value = 'male', inplace = True)

    if str.lower(col.Gender) in trans_str:
        data['Gender'].replace(to_replace = col.Gender, value = 'trans', inplace = True)

    if str.lower(col.Gender) in female_str:
        data['Gender'].replace(to_replace = col.Gender, value = 'female', inplace = True)

random_list = ['A little about you', 'p']
data = data[~data['Gender'].isin(random_list)]

print(data['Gender'].unique())

data['Age'].fillna(data['Age'].median(), inplace = True)

s = pd.Series(data['Age'])
s[s<18] = data['Age'].median()
data['Age'] = s
s = pd.Series(data['Age'])
s[s>120] = data['Age'].median()
data['Age'] = s

data['age_range'] = pd.cut(data['Age'], [0,20,30,65,100], labels = ["0-20", "21-30", "31-65", "66-100"], include_lowest = True)

data['self_employed'] = data['self_employed'].replace([defaultString], 'No')
print(data['self_employed'].unique())

data['work_interfere'] = data['work_interfere'].replace([defaultString], 'Don\'t know')
print(data['work_interfere'].unique())

data['Gender'].unique()

data['Age'].unique()

data['family_history'].unique()

data['work_interfere'].unique()

labelDict = {}
for feature in data:
    le = preprocessing.LabelEncoder()
    le.fit(data[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    data[feature] = le.transform(data[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue

# for key, value in labelDict.items():
#     print(key, value)

data.head()

total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
print(missing_data)

#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

#treatment correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'treatment')['treatment'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(data["Age"], bins=24)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")

# define X and y
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
X = data[feature_cols]
y = data.treatment

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Create dictionaries for final graph
methodDict = {}
rmseDict = ()

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest.fit(X, y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]

labels = []

for f in range(X.shape[1]):
    labels.append(feature_cols[f])

#Plotting the feature importance of the forest
plt.figure(figsize=(12,8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()

def evalClassModel(model, y_test, y_pred_class, plot=False):

    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    print('Null accuracy:\n', y_test.value_counts())

    
    print('Percentage of ones:', y_test.mean())

    print('Percentage of zeros:',1 - y_test.mean())

    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])

    confusion = metrics.confusion_matrix(y_test, y_pred_class)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    sns.heatmap(confusion,annot=True,fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    accuracy = metrics.accuracy_score(y_test, y_pred_class)
    print('Classification Accuracy:', accuracy)

    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred_class))

    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)

    print('Precision:', metrics.precision_score(y_test, y_pred_class))

    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))

    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())


    model.predict_proba(X_test)[0:10, 1]

    y_pred_prob = model.predict_proba(X_test)[:, 1]

    if plot == True:
        # histogram of predicted probabilities
        # adjust the font size
        plt.rcParams['font.size'] = 12
        # 8 bins
        plt.hist(y_pred_prob, bins=8)

        # x-axis limit from 0 to 1
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')


    y_pred_prob = y_pred_prob.reshape(-1,1)
    y_pred_class = binarize(y_pred_prob)[0]


    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()

        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()

    def evaluate_threshold(threshold):
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])

    return accuracy

# # Logistic Regression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print('########### Logistic Regression ###############')
accuracy_score = evalClassModel(lr, y_test, y_pred, True)

methodDict['Log. Regres.'] = accuracy_score * 100

# # RFC

random_forest = RandomForestClassifier(max_depth = None, min_samples_leaf=8, min_samples_split=2, n_estimators = 20, random_state = 1)
forest = random_forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

print('########### Random Forest ###############')
accuracy_score = evalClassModel(forest, y_test, y_pred, True)

methodDict['R. Forest'] = accuracy_score * 100

# # KNN

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('########### KNeighborsClassifier ###############')
accuracy_score = evalClassModel(knn, y_test, y_pred, True)

methodDict['KNN'] = accuracy_score * 100

# # Naive Bayes

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print('########### Naive Bayes Classifier ###############')
accuracy_score = evalClassModel(gnb, y_test, y_pred, True)

methodDict['NB'] = accuracy_score * 100

# # Conclusion 

def plotSuccess():
    s = pd.Series(methodDict)
    s = s.sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    #Colors
    ax = s.plot(kind='bar')
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylim([70.0, 90.0])
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Success of methods')

    plt.show()
    
plotSuccess()

