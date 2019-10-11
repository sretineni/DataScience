#!/usr/bin/env python
# coding: utf-8

# # Homework 2 - IEEE Fraud Detection

# For all parts below, answer all parts as shown in the Google document for Homework 2. Be sure to include both code that justifies your answer as well as text to answer the questions. We also ask that code be commented to make it easier to follow.

# In[148]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sbn
sbn.set(style="darkgrid")
import matplotlib.mlab as mat
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from collections import Counter


# In[246]:


#load data
#take necessary columns from identity table of train data
df_train_id = pd.read_csv("C:/Users/sneha/Downloads/Homeworks/DSF/train_identity.csv")
df_train_idsel = df_train_id[['TransactionID','DeviceType','DeviceInfo']]
df_train_idsel.head()

#take necessary columns from transaction table of train data
df_train_trans = pd.read_csv("C:/Users/sneha/Downloads/Homeworks/DSF/train_transaction.csv") 
df_train_transsel = df_train_trans[['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD','card4','card6','P_emaildomain','R_emaildomain',
'addr1','addr2','dist1','dist2']]
df_train_transsel.head()


# In[247]:


#Outer join on both transaction and identity tables of train data
df_c = pd.merge(df_train_idsel, df_train_transsel, on='TransactionID', how='outer')
df_c.head()


# ## Part 1 - Fraudulent vs Non-Fraudulent Transaction

# DEVICE TYPE distribution for fraudulent and non fraudulent transactions.

# In[248]:


# TODO: code and runtime results
# We filter out the fraudulent and non fraudulent transactions.
df_fraud = df_c[df_c['isFraud']==1]
df_nfraud = df_c[df_c['isFraud']==0]


# In[462]:


#We plot the percentage of each device type for fraudulent and non fraudulent transactions.
fg, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
title = ['Device type for Fraudulent' , 'Device type for Non Fraudulent']
data = [df_fraud['DeviceType'].value_counts(normalize=True) * 100, df_nfraud['DeviceType'].value_counts(normalize=True) * 100]
for i in range(0,len(data)):
    data[i].plot(kind='bar', ax=ax[i])
    ax[i].title.set_text(title[i])


# We can observe that fraudulent transactions have almost the same percentage of mobile and desktop devices, 
# while non fraudulent ones mostly used desktop devices.

# Percentage distribution of fraudulent and non fraudulent transactions for each card type in card4.

# In[360]:


#CARD4 - We find out the percentage of fraudulent transactions of each card type.
card4Tab = pd.crosstab(index=df_c["card4"],columns=df_c["isFraud"], normalize='index')
card4Tab = card4Tab*100
print(card4Tab)


# In[467]:


#Plotting the percentages of fraudulent and non fraudulent transactions of each card type. 
plot = card4Tab.plot(kind="bar",figsize=(6,5),stacked=True)
plot.set_title('% Fraudulent/Non Fraudulent Vs Card4 type')


# We observe that discover card has the highest percentage of fraudulent transactions, at almost 8%, 
# while American Express has the least percentage of fraudulent transactions.

# Percentage distribution of fraudulent and non fraudulent transactions for each card type in card6.

# In[253]:


#CARD6 - We find out the percentage of fraudulent transactions of each card type in card6.
card6Tab = pd.crosstab(index=df_c["card6"],columns=df_c["isFraud"], normalize='index')
card6Tab = card6Tab*100
print(card6Tab)


# In[468]:


#Plotting the percentages of fraudulent and non fraudulent transactions of each card6 type. 

card6plot = card6Tab.plot(kind="bar",figsize=(8,5),stacked=True)
card6plot.set_title('% Fraudulent/Non Fraudulent Vs Card6 type')


# We observe that credit cards have the highest percentage of fraudulent transactions, followed by debit. 
# Interestingly, charged card had no fraudulent transactions.

# Percentage distribution of fraudulent and non fraudulent transactions for each product code.

# In[255]:


#ProductCD We find out the percentage of fraudulent transactions of each PRODUCT CODE.
ProductCDTab = pd.crosstab(index=df_c["ProductCD"],columns=df_c["isFraud"], normalize='index')
ProductCDTab = ProductCDTab*100
print(ProductCDTab)


# In[469]:


productplot = ProductCDTab.plot(kind="bar",figsize=(8,5),stacked=True)
productplot.set_title('% Fraudulent/Non Fraudulent Vs product code')


# Product code C has the highest, i.e, 11.7% fraudulent transactions and Product code W has the least fraudulent transactions.

# Percentage distribution of fraudulent and non fraudulent transactions for each purchaser email domain.

# In[257]:


#The P_emaildomain feature has a lot of unique values. So, we take the email domains whose count is less than 5% of the total rows and name them 'P_EmaildomainOther'

df_c.P_emaildomain[df_c.P_emaildomain.replace(Counter(df_c.P_emaildomain)) < len(df_c.P_emaildomain)*0.05] = 'P_emaildomainOther'


# In[258]:


#P_emaildomain - We find out the percentage of fraudulent transactions of each Purchaser email domain.
P_emaildomainTab = pd.crosstab(index=df_c["P_emaildomain"],columns=df_c["isFraud"], normalize='index')
P_emaildomainTab = P_emaildomainTab*100
print(P_emaildomainTab)


# In[470]:


pplot = P_emaildomainTab.plot(kind="bar",figsize=(8,5),stacked=True)
pplot.set_title('% Fraudulent/Non Fraudulent Vs  P email domain')


# Hotmail has the highest percentage of fraudulent transactions, at 5.3%. 

# In[259]:


#The R_emaildomain feature has a lot of unique values. So, we take the email domains whose count is less than 1% of the total rows and name them 'R_EmaildomainOther'

df_c.R_emaildomain[df_c.R_emaildomain.replace(Counter(df_c.R_emaildomain)) < len(df_c.R_emaildomain)*0.01] = 'R_emaildomainOther'


# In[260]:


#R_emaildomain - We find out the percentage of fraudulent transactions of each Receiver email domain.
R_emaildomainTab = pd.crosstab(index=df_c["R_emaildomain"],columns=df_c["isFraud"], normalize='index')
R_emaildomainTab = R_emaildomainTab*100
print(R_emaildomainTab)


# In[472]:


rplot = R_emaildomainTab.plot(kind="bar",figsize=(8,5),stacked=True)
rplot.set_title('% Fraudulent/Non Fraudulent Vs  R email domain')


# The percentage of fraudulent transactions are more with gmail.com as receiver address.

# In[262]:


#In the DeviceInfo column, we take the devices whose count is less than 1% of the total rows and name them 'DeviceInfoOther'

df_c.DeviceInfo[df_c.DeviceInfo.replace(Counter(df_c.DeviceInfo)) < len(df_c.DeviceInfo)*0.01] = 'DeviceInfoOther'


# In[263]:


#DeviceInfo - We find out the percentage of fraudulent transactions of each major device info.
DeviceInfoTab = pd.crosstab(index=df_c["DeviceInfo"],columns=df_c["isFraud"], normalize='index')
DeviceInfoTab = DeviceInfoTab*100
print(DeviceInfoTab)


# In[473]:


deviceInfoplot = DeviceInfoTab.plot(kind="bar",figsize=(8,5),stacked=True)
deviceInfoplot.set_title('% Fraudulent/Non Fraudulent Vs  Device Info')


# Windows and IOS devices as a single device info have the highest percentage of fraudulent transactions 
# at 6.5 and 6.3% respectively.

# Distribution of transaction amount over fraudulent transactions.

# In[474]:


plt.figure(figsize=(10,10))
#sbn.distplot(df_nfraud['TransactionAmt'],bins=100)
Tamtplot = sbn.distplot(df_fraud['TransactionAmt'],bins=100)
Tamtplot.set_title('TransactionAmt Vs Frequency of Fraudulent transactions')


# Most fraudulent transactions happen over smaller amounts, mostly between 0 to 300. There is again a small spike above 5000.

# In[230]:


df_nfraud['TransactionAmt'].describe()


# Percentage distribution of fraudulent transactions with TransactionDT

# We initially perform preprocessing on the TransactionDT column. 

# In[457]:


#The column seems to denote time delta in seconds. So, we add a column denoting the time delta in number of days.
totalDays = (df_c['TransactionDT'].max()-df_c['TransactionDT'].min())/86400
#totalDays = 181.9 days
#Marking day numbers for the transactions
df_c['TransactionDTDayNumber'] = df_c['TransactionDT'].apply(lambda x: x/86400).astype(int)
df_c.head()


# In[477]:


plt.figure(figsize=(20,8))
daynumVStransaction = sbn.countplot(x="TransactionDTDayNumber", data=df_c)
for idx, label in enumerate(daynumVStransaction.get_xticklabels()):
    if idx % 10 == 0:  # every 10th label is retained.
        label.set_visible(True)
    else:
        label.set_visible(False)
daynumVStransaction.set_title('Frequency of Transactions VS TransactionDate')


# There is a sudden spike in transactions on the 25th day. From the visualisation in cell _ below, there is a spike on 
# day 390 in the test data as well. There is a gap of 31 days between the train and test data. This means, the test data starts
# one month after train data ends. Also, 390-25 = 365, which means they're exactly one year apart. Assuming these 2 days are 
# Christmas,we fit this assumption and the start date comes to be December 1. I researched on kaggle to find similar assumptions and found one, where the year was taken as 2017. Referred from: https://kaggle.com/c/ieee-fraud-detection/discussion/100071#latest-604196

# Plot of count of fraudulent transactions vs Transaction day number.

# In[478]:


plt.figure(figsize=(20,5))
daynumVStransactionFraud = sbn.countplot(x="TransactionDTDayNumber", data=df_c[df_c['isFraud']==1])
for idx, label in enumerate(daynumVStransactionFraud.get_xticklabels()):
    if idx % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
daynumVStransactionFraud.set_title('Frequency of Fraudulent Transactions VS TransactionDate')


# In[480]:


plt.figure(figsize=(20,5))
daynumVStransactionNFraud = sbn.countplot(x="TransactionDTDayNumber", data=df_c[df_c['isFraud']==0])
for idx, label in enumerate(daynumVStransactionNFraud.get_xticklabels()):
    if idx % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
daynumVStransactionNFraud.set_title('Frequency of Non Fraudulent Transactions VS TransactionDate')


# In[268]:


#Guessing start date to be Dec 1 2017, we find the date of every transaction.
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
df_c['TransactionDTDate'] = df_c['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

print(df_c['TransactionDTDate'].head())
print(df_c['TransactionDTDate'].tail())


# ## Part 2 - Transaction Frequency

# From the above transaction date, we calculate the hour of day, month and day of the week. This data can be used to find 
# correlations with isFraud during further analysis.

# In[380]:


#Finding day of the week
df_c['DayOfTheWeek'] = df_c['TransactionDTDate'].dt.weekday_name
df_c.head()


# In[272]:


#Finding hour of day
df_c['hourOfDay'] = df_c.TransactionDTDate.dt.hour
df_c.head()
#df_cumulative.tail()


# In[273]:


#Finding month
df_c['month'] = df_c.TransactionDTDate.dt.month
df_c.head()
#df_cumulative.tail()


# In[459]:


#We find the most frequent country code. Turns out its 87.
df_cumulative['addr2'].value_counts()


# In[481]:


plt.figure(figsize=(20,5))
hourofdayVStransactionfrequency = sbn.countplot(x="hourOfDay", data=df_c[df_c['addr2']==87])
hourofdayVStransactionfrequency.set_title('Hour of Day VS Frequency of transactions')


# In the graph, between hour 6 and hour 10, the transactions are the lowest. They most probably should be the sleeping hours in country 87. If we consider 8 am to 12pm to be the waking hours, we can map these timings approximately to 12 pm and 4 am in the reference time. The trend of transactions gradually increase from 12pm reference, as during the day, stay consistent for a period of time(which is presumably the evening) and reduce as the night hours increase. This also supports our observation. So, the country with code 87 is roughly 4 hours behind the reference time.

# ## Part 3 - Product Code

# To understand the distribution of transaction amount with product codes, we plot a boxplot first.

# In[275]:


# TODO: code to analyze prices for different product codes
plt.figure(figsize=(30,5))
productcodeVStransAmt = sbn.catplot(x='ProductCD',y='TransactionAmt',kind='box', data=df_c)
productcodeVStransAmt 


# Product code W seems to have outliers, so we check the number of records with abnormally high transaction amount.

# In[276]:


#Only 2 rows with transaction amount > 10000, both correspond to product code W.
df_c[df_c['TransactionAmt']>10000]


# In[483]:


#We visualise the data distributions of transaction amounts < 10000.
df_c_below10k = df_c[df_c['TransactionAmt']<10000]
plt.figure(figsize=(30,15))
productcodeVStransAmt10k = sbn.catplot(x='TransactionAmt',y='ProductCD',kind='box', data=df_c_below10k, height=10, aspect=2)
productcodeVStransAmt10k


# Product code W has a lot of high amount transactions, but the 25th, 50th, 75th percentile of R product code are higher 
# than that of W. We now examine the mean transaction amount of each product code.

# In[131]:


print("W - ",df_c[df_c['ProductCD']=='W']['TransactionAmt'].mean())
print("H - ",df_c[df_c['ProductCD']=='H']['TransactionAmt'].mean())
print("C - ",df_c[df_c['ProductCD']=='C']['TransactionAmt'].mean())
print("S - ",df_c[df_c['ProductCD']=='S']['TransactionAmt'].mean())
print("R - ",df_c[df_c['ProductCD']=='R']['TransactionAmt'].mean())


#Max is R and min is C


# R is the product code that refers to the most expensive products and C refers to the least expensive products.

# ## Part 4 - Correlation Coefficient

# We now plot the bar graph between the time of day (hour of day in this case) and avg transaction amount.

# In[218]:


df_c['hourOfDay'].value_counts()


# In[484]:


# TODO: code to calculate correlation coefficient
# Plot purchase amount mean VS hour of day
plt.figure(figsize=(20,5))
Plot_purchaseAmtVShourOfDay = sbn.barplot(x="hourOfDay", y="TransactionAmt", data=df_c)
Plot_purchaseAmtVShourOfDay.set_title('Time of day VS transaction Amt')


# In[220]:


# Calculatine correlation coefficient between hour of day and transaction amount.
df_c['hourOfDay'].corr(df_c['TransactionAmt'])


# In[490]:


df_c.corr()


# The correlation is 0.045.

# ## Part 5 - Interesting Plot

# In[485]:


# TODO: code to generate the plot here.
plt.figure(figsize=(20,5))
daynumVStransactionFraud = sbn.countplot(x="TransactionDTDayNumber", data=df_c[df_c['isFraud']==1])
for idx, label in enumerate(daynumVStransactionFraud.get_xticklabels()):
    if idx % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
daynumVStransactionFraud.set_title('Frequency of fraudulent transactions VS Transaction date')


# The fraudulent transactions take a sudden noticeable spike on day 64. 64 days after Dec 1 is Feb 3 2018. 
# Upon researching on the internet, I found that Feb 8 2013 was the day Dow Jones Index crashed 665 points, making it the worst 
# in two years. Referred from: [Link](https://theweek.com/10things/750984/10-things-need-know-today-february-3-2018). So, a large
# number of fraudulent transactions(mostly of smaller value) may be done on that day in the form of selling shares in anticipation of the decline in prices.

# ## Part 6 - Prediction Model

# For building the prediction model, we merge train and test data and perform preprocessing.

# In[286]:


# TODO: code for your final model
#Outer join on both transaction and identity tables of train data
c_copy = pd.merge(df_train_idsel, df_train_transsel, on='TransactionID', how='outer')
c_copy.head()


# In[287]:


Xtrain = c_copy.drop(['isFraud'], axis=1)
Xtrain.shape


# In[289]:


Ytrain = c_copy['isFraud']
Ytrain.shape


# In[293]:


#Import test data
#take necessary columns from identity table of test data
df_test_id = pd.read_csv("C:/Users/sneha/Downloads/Homeworks/DSF/test_identity.csv")
df_test_idsel = df_test_id[['TransactionID','DeviceType','DeviceInfo']]
df_test_idsel.head()


# In[294]:


df_test_idsel.shape


# In[295]:


#take necessary columns from transaction table of test data
df_test_trans = pd.read_csv("C:/Users/sneha/Downloads/Homeworks/DSF/test_transaction.csv") 
df_test_transsel = df_test_trans[['TransactionID','TransactionDT','TransactionAmt','ProductCD','card4','card6','P_emaildomain','R_emaildomain',
'addr1','addr2','dist1','dist2']]
df_test_transsel.shape


# In[296]:


#Outer join on both transaction and identity tables of test data
c1 = pd.merge(df_test_idsel, df_test_transsel, on='TransactionID', how='outer')
c1.head()
c1.shape


# In[460]:


#Pre processing on the TransactionDT column
df_c2 = df_c1.copy()
totalDays = (df_c2['TransactionDT'].max()-df_c2['TransactionDT'].min())/86400
#totalDays = 181.9 days
#Marking day numbers for the transactions
df_c2['TransactionDTDayNumber'] = df_c2['TransactionDT'].apply(lambda x: x/86400).astype(int)
df_c2.head()


# Distribution of transactions with TransactionDate. Spike on day 390. Used this previously in the analysis to support 
# the choice of dec 1 2017 as start date.

# In[486]:


plt.figure(figsize=(20,5))
daynumVStransaction = sbn.countplot(x="TransactionDTDayNumber", data=df_c2)
for idx, label in enumerate(daynumVStransaction.get_xticklabels()):
    if idx % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
daynumVStransaction.set_title('Frequency of transactions VS Transaction date')


# In[297]:


Xtest = c1
Xtest.shape


# In[298]:


#append Xtrain and Xtest into a new dataframe called df_cumulative
df_cumulative = Xtrain.append(Xtest)
df_cumulative.shape


# Preprocessing on date column

# In[300]:


# Preprocessing for date column
totalDays = (df_cumulative['TransactionDT'].max()-df_cumulative['TransactionDT'].min())/86400
totalDays


# In[461]:


#Marking day numbers for the transactions
df_cumulative['TransactionDTDayNumber'] = df_cumulative['TransactionDT'].apply(lambda x: x/86400).astype(int)
df_cumulative.head()


# In[303]:


#Guessing start date to be Dec 1 2017, we find the date of every transaction.
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
df_cumulative['TransactionDTDate'] = df_cumulative['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

print(df_cumulative['TransactionDTDate'].head())
print(df_cumulative['TransactionDTDate'].tail())


# In[304]:


df_cumulative['DayOfTheWeek'] = df_cumulative['TransactionDTDate'].dt.weekday_name
df_cumulative.head()


# We change the email domains and device infos which are less than 5% of the data to 'other' type.

# In[306]:


#In the P_emaildomain column, we take the email domains whose count is less than 5% of the total rows and name them 'P_EmaildomainOther'
df_cumulative.P_emaildomain[df_cumulative.P_emaildomain.replace(Counter(df_cumulative.P_emaildomain)) < len(df_cumulative.P_emaildomain)*0.05] = 'P_emaildomainOther'


# In[308]:


#In the R_emaildomain column, we take the email domains whose count is less than 5% of the total rows and name them 'R_EmaildomainOther'
df_cumulative.R_emaildomain[df_cumulative.R_emaildomain.replace(Counter(df_cumulative.R_emaildomain)) < len(df_cumulative.R_emaildomain)*0.05] = 'R_emaildomainOther'


# In[310]:


#In the DeviceInfo column, we take the devices whose count is less than 5% of the total rows and name them 'DeviceInfoOther'
df_cumulative.DeviceInfo[df_cumulative.DeviceInfo.replace(Counter(df_cumulative.DeviceInfo)) < len(df_cumulative.DeviceInfo)*0.05] = 'DeviceInfoOther'


# In[311]:


df_cumulative[df_cumulative['DeviceInfo']=="DeviceInfoOther"]['TransactionID'].count()


# In[312]:


#Count null values of each feature
df_cumulative.isna().sum()


# To fill null values of R_emaildomain, we analyse the P_emaildomain and dist1(some form of distance between the purchaser and receiver).
# For entries with a non null P_emaildomain and dist1=0, we fill the null value of R_emaildomain with P_emaildomain itself, 
# since the purchaser and receiver will probably have the same email address. 

# In[313]:


df_cumulative[df_cumulative['dist1']==0].count()


# In[314]:


df_cumulative[(df_cumulative['dist1']==0) & (df_cumulative['P_emaildomain'].notnull()) ].count()


# In[315]:


# We replace the R_emaildomain with the P_emaildomain for entries with dist1 = 0.
#Since we assume dist1 is some form of distance between purcjaser and receiver.
df_cumulative['R_emaildomain'] = np.where((df_cumulative['dist1']==0) & (df_cumulative['P_emaildomain'].notnull()) , df_cumulative['P_emaildomain'],df_cumulative['R_emaildomain'])


# Filling the null values. We can replace missing categorical values to 'unknown' type, mainly because in transaction data, 
# null value may be hiding or masking something. So, we can find the correlation with the unknown type for any interesting observations.

# In[316]:


#replacing missing device type values to unkdevicetype
df_cumulative.DeviceType.fillna('unkDeviceType',inplace=True)
df_cumulative.DeviceType.value_counts()


# In[317]:


#replacing missing device info values to unkdeviceinfo
df_cumulative.DeviceInfo.fillna('unkDeviceInfo',inplace=True)
df_cumulative.DeviceInfo.value_counts()


# In[318]:


#df_cumulative['card6'].isnull().sum() = 1571. So, we replace it with unkcard6 type
df_cumulative.card6.fillna('unkcard6',inplace=True)
df_cumulative.card6.value_counts()


# In[319]:


#df_cumulative['card4'].isnull().sum() = 1577. So, we replace it with unkcard4 type
df_cumulative.card4.fillna('unkcard4',inplace=True)
df_cumulative['card4'].value_counts()


# In[320]:


#df_cumulative['addr2'].isnull().sum() = 65706. We replace with unkaddr2 type.
df_cumulative.addr2.fillna('unkaddr2',inplace=True)
df_cumulative['addr2'].value_counts()


# In[321]:


#df_cumulative['addr1'].isnull().sum() = 65706. We replace with unkaddr1 type.
df_cumulative.addr1.fillna('unkaddr1',inplace=True)
df_cumulative['addr1'].value_counts()


# In[322]:


#df_cumulative['P_emaildomain'].isnull().sum() = 94456. Replace with unkPemail
df_cumulative.P_emaildomain.fillna('unkPemail',inplace=True)
df_cumulative['P_emaildomain'].value_counts()


# In[323]:


#df_cumulative['R_emaildomain'].isnull().sum() = 438670. Replace with unkRemail
df_cumulative.R_emaildomain.fillna('unkRemail',inplace=True)
df_cumulative['R_emaildomain'].value_counts()


# In[324]:


df_cumulative.isnull().sum()


# In[325]:


df_cumulative['hourOfDay'] = df_cumulative.TransactionDTDate.dt.hour
df_cumulative.head()
#df_cumulative.tail()


# One hot encoding.

# In[326]:


#Performing one hot encoding on the dataframe
df_cumulative_ohe = pd.get_dummies(df_cumulative)
df_cumulative_ohe.shape


# In[327]:


df_c_copy = df_cumulative_ohe.copy()
df_c_copy = df_c_copy.drop(['dist1','dist2','TransactionID','TransactionDT','TransactionDTDate'], axis=1)


# Splitting the train and test data for model building.

# In[332]:


trainSize = Xtrain.shape[0]
testSize = Xtest.shape[0] 
print("test data size is",testSize)
print("train data size is",trainSize)


# In[334]:


#dfSize = df_c_copy.shape[0] = 1097231
Xtr = df_c_copy[0:trainSize]
Xtr.shape


# In[355]:


Xte = df_c_copy[dfSize-testSize:dfSize]
Xte.head()


# Building a decision tree classifier.

# In[399]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = Xtr
y = Ytrain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[443]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
prediction1 = classifier.predict(X_test)


# Finding out accuracy.

# In[444]:


print("\nThe classifier's accuracy is %s percent\n" % round(100*accuracy_score(y_test, prediction1), 2))


# Calculating the AUC score

# In[445]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[449]:


probs = classifier.predict_proba(X_test)
probs = probs[:, 1]


# In[450]:


auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# Plotting the ROC curve

# In[453]:



def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# Reference: [https://stackabuse.com/understanding-roc-curves-with-python/]

# In[488]:


fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(5,5))
plot_roc_curve(fpr, tpr)


# Making predictions.

# In[434]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

classifier = DecisionTreeClassifier()
classifier.fit(Xtr,Ytrain)
prediction1 = classifier.predict(Xte)


# In[436]:


df_pred.to_csv('C:\\Users\\sneha\\Downloads\\Homeworks\\DSF\\prediction_final.csv', index=False)


# ## Part 7 - Final Result

# Report the rank, score, number of entries, for your highest rank. Include a snapshot of your best score on the leaderboard as confirmation. Be sure to provide a link to your Kaggle profile. Make sure to include a screenshot of your ranking. Make sure your profile includes your face and affiliation with SBU.

# Kaggle Link: https://www.kaggle.com/snehageetharsg

# Highest Rank: 5715

# Score: 0.5714

# Number of entries: 2

# INCLUDE IMAGE OF YOUR KAGGLE RANKING!![KaggleRank.png](attachment:KaggleRank.png)

# In[ ]:




