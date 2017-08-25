# Project Identify Fraud from Enron Email

## Project Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I built a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal. This data was combined with a hand-generated list of POI in the fraud case.

## Project question and dataset 

##### Goal of the project
The goal of the project is to determine who is a POI based on financial and email data made public.
Machine learning in this case can be useful by classifying persons based on the financial and email data into a POI group and a non-POI group.
For this purpose, POI were identified based on the fact if they were indicted, reached a settlement or plea deal or testified in exchange for prosecution immunity

The dataset provided contains 146 datapoints and 21 features (14 financial, 6 email and 1 POI feature).
From the 146 datapoints, 18 are marked as POI and 128 are marked as non-POI.
Not all datapoints have values for every feature:


</div>
<table class="tg">
  <tr>
    <th class="tg-9hbo">Feature</th>
    <th class="tg-amwm">Count</th>
  </tr>
  <tr>
    <td class="tg-yw4l">bonus</td>
    <td class="tg-baqh">82</td>
  </tr>
  <tr>
    <td class="tg-yw4l">deferral_payments</td>
    <td class="tg-baqh">39</td>
  </tr>
  <tr>
    <td class="tg-yw4l">deferred_income</td>
    <td class="tg-baqh">49</td>
  </tr>
  <tr>
    <td class="tg-yw4l">director_fees</td>
    <td class="tg-baqh">17</td>
  </tr>
  <tr>
    <td class="tg-yw4l">email_address</td>
    <td class="tg-baqh">11</td>
  </tr>
  <tr>
    <td class="tg-yw4l">exercised_stock_options</td>
    <td class="tg-baqh">102</td>
  </tr>
  <tr>
    <td class="tg-yw4l">expenses</td>
    <td class="tg-baqh">95</td>
  </tr>
  <tr>
    <td class="tg-yw4l">from_messages</td>
    <td class="tg-baqh">86</td>
  </tr>
  <tr>
    <td class="tg-yw4l">from_poi_to_this_person</td>
    <td class="tg-baqh">86</td>
  </tr>
  <tr>
    <td class="tg-yw4l">from_this_person_to_poi</td>
    <td class="tg-baqh">86</td>
  </tr>
  <tr>
    <td class="tg-yw4l">loan_advances</td>
    <td class="tg-baqh">4</td>
  </tr>
  <tr>
    <td class="tg-yw4l">long_term_incentive</td>
    <td class="tg-baqh">66</td>
  </tr>
  <tr>
    <td class="tg-yw4l">other</td>
    <td class="tg-baqh">93</td>
  </tr>
  <tr>
    <td class="tg-yw4l">poi</td>
    <td class="tg-baqh">146</td>
  </tr>
  <tr>
    <td class="tg-yw4l">restricted_stock</td>
    <td class="tg-baqh">110</td>
  </tr>
  <tr>
    <td class="tg-yw4l">restricted_stock_deferred</td>
    <td class="tg-baqh">18</td>
  </tr>
  <tr>
    <td class="tg-yw4l">salary</td>
    <td class="tg-baqh">95</td>
  </tr>
  <tr>
    <td class="tg-yw4l">shared_receipt_with_poi</td>
    <td class="tg-baqh">86</td>
  </tr>
  <tr>
    <td class="tg-yw4l">to_messages</td>
    <td class="tg-baqh">86</td>
  </tr>
  <tr>
    <td class="tg-yw4l">total_payments</td>
    <td class="tg-baqh">125</td>
  </tr>
  <tr>
    <td class="tg-yw4l">total_stock_value</td>
    <td class="tg-baqh">126</td>
  </tr>
</table>
</div>


## Features

While exploring the dataset I noticed the following:
1) Two odd POI, namely 'THE TRAVEL AGENCY IN THE PARK' with only a value for the features other and total, and 'TOTAL' 
with very high values (outliers) from which i assume it is the total of all data points. Since these don't represent
persons, I removed them from the dataset. 
2) LOCKHART EUGENE E is the only person with NaN values for all features. I removed him from the dataset.

I used SelectKBest to determine the most powerful features for classifying after rescaling them with the MinMax rescaler.
I made the assumption that POI share fraud related information more between themselves then with non-POI and that this
could show through the relative portion of mail they send and receive from other POI. For this I created four additional
features, from which from_poi_perc had the highest score. 


<table class="tg">
  <tr>
    <th class="tg-yw4l">result SelectKBest</th>
    <th class="tg-yw4l">score</th>
  </tr>
  <tr>
    <td>exercised_stock_options</td>
    <td>24.82</td>
  </tr>
  <tr>
    <td>total_stock_value</td>
    <td>24.18</td>
  </tr>
  <tr>
    <td>bonus</td>
    <td>20.79</td>
  </tr>
  <tr>
    <td>salary</td>
    <td>18.29</td>
  </tr>
   <tr>
    <td>deferred_income</td>
    <td>11.46</td>
  </tr>
  <tr>
    <td>long_term_incentive</td>
    <td>9.92</td>
  </tr>
  <tr>
    <td>restricted_stock</td>
    <td>9.21</td>
  </tr>
  <tr>
    <td>total_payments</td>
    <td>8.77</td>
  </tr>
  <tr>
    <td>shared_receipt_with_poi</td>
    <td>8.59</td>
  </tr>
 <tr>
    <td>from_poi_perc</td>
    <td>7.22</td>
  </tr>
</table>


## Picking an algorithm

I looked at the following classification algorithms, with and without PCA. Principal component analysis is used on all these algorithms to reduce the dimensionality of the input features.
With 5 principal components I get 92% of the variation in the data.
<TABLE BORDER=0>
<TR>
<TD>
<table>
<tr>
    <th>Gaussian Naive Bayes  with PCA</th>
    <th></th>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.84607</td>
  </tr>
  <tr>
    <td>precision</td>
    <td bgcolor="#FFFF00"> 0.41571</td>
  </tr>
  <tr>
    <td>recall</td>
    <td bgcolor="#FFFF00">0.381</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.3976</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Decision Tree Classifier  with PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.79153</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.23482</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.2495</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.24194</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Support Vector Machines  with PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.86327</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.05263</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.0015</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.00292</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Logistic Regression  with PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.868</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.54386</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.062</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.11131</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>KMeans  with PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.6998</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.18033</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.353</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.23872</td>
  </tr>
</table>
</TD>
<TD>
<table>
  <tr>
    <th>Gaussian Naive Bayes  without PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.83567</td>
  </tr>
  <tr>
    <td>precision</td>
    <td bgcolor="#FFFF00">0.3607</td>
  </tr>
  <tr>
    <td>recall</td>
    <td bgcolor="#FFFF00">0.301</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.32815</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Decision Tree Classifier  without PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.81233</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.29325</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.289</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.29111</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Support Vector Machines  without PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.86353</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.07273</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.002</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.00389</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Logistic Regression  without PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.87453</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.68098</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.111</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.19089</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>KMeans  without PCA</td>
    <td></td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td>0.6748</td>
  </tr>
  <tr>
    <td>precision</td>
    <td>0.17266</td>
  </tr>
  <tr>
    <td>recall</td>
    <td>0.3795</td>
  </tr>
  <tr>
    <td>f1</td>
    <td>0.23734</td>
  </tr>
</table>  
</TD>
</TR>
</TABLE>                

Gaussian Naive Bayes, with and without PCA, is the only algorithm that scores more than 0.3 for both recall and
precision.

Gaussian Naive Bayes is a simple algorithm that can't be tuned, I therefore tried to see if tuning the second best, 
decision tree, would result in better scores.

## Tuning an algorithm

Parameter tuning in machine learning is the process of optimizing parameter settings for a learning algorithm.
Tuning your algorithm allows you to get the best possible results. 

@@@@@@@
I used GridSearchCV with the following parameters to tune DecisionTreeClassifier: 
{'n_clusters': [1,2,3,4,5,6,7,8,9] , 'algorithm':('auto', 'full', 'elkan')
                , 'init':('k-means++', 'random'), 'n_init': [1,2,5,10,20]}.
                
The best estimator was KMeans(algorithm='full', copy_x=True, init='random', max_iter=300,
    n_clusters=9, n_init=2, n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001, verbose=0).
     
I noticed GridSearchCV uses a different scoring and doesn't follow my high recall on poi/high precision on non-poi
scoring. I lowered the n_clusters step by step to find an optimum and found that at n_clusters = 2. 
Then, the best estimator was KMeans(algorithm='elkan', copy_x=True, init='random', max_iter=300,
    n_clusters=2, n_init=1, n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001, verbose=0).    
@@@@@@


<TABLE BORDER=0>
<TR>
<TD>
<table>
  <tr>
    <th>Gaussian Naive Bayes  with PCA</th>
    <th></th>
  </tr>
  <tr>
    <td>precision poi</td>
    <td>0.41</td>
  </tr>
  <tr>
    <td>precision non-poi</td>
    <td>0.9</td>
  </tr>
  <tr>
    <td>recall poi</td>
    <td>0.33</td>
  </tr>
  <tr>
    <td>recall non-poi</td>
    <td>0.92</td>
  </tr>
  <tr>
    <td>f1 poi</td>
    <td>0.34</td>
  </tr>
  <tr>
    <td>f1 non-poi</td>
    <td>0.91</td>
  </tr>
</table>
</TD>
<TD> 
<table>
  <tr>
    <th>Tuned KMeans without PCA</th>
    <th></th>
  </tr>
  <tr>
    <td>precision poi</td>
    <td>0.26</td>
  </tr>
  <tr>
    <td>precision non-poi</td>
    <td>0.74</td>
  </tr>
  <tr>
    <td>recall poi</td>
    <td>0.47</td>
  </tr>
  <tr>
    <td>recall non-poi</td>
    <td>0.55</td>
  </tr>
  <tr>
    <td>f1 poi</td>
    <td>0.24</td>
  </tr>
  <tr>
    <td>f1 non-poi</td>
    <td>0.58</td>
  </tr>
</table>  
</TD>
</TR>
</TABLE>                


## Validation

Model validation in machine learning is the process where a trained model is evaluated with a testing data set, where 
training and testing data are separate portions of the same data set. We do this to make sure that the model has the
ability to be used to make predictions with a known reliability using other data sets.

I validated my results originally by splitting the data in 2/3 of training data and 1/3 of test data. To get better
metrics, I repeatedly fitted my classifiers and averaged the metrics over all the fits. But while doing this I did not
take into account that each fitting did not have approximately the same number of data points of POI and non-POI. 
I solved this by using the stratified shuffle split with the same settings as tester.py, to be able to compare the
results. This prevents imbalance in the distribution of the POI-indicator in the training and testing set. 

## Evaluation Metrics

Precision and recall were used as evaluation metrics. Recall is a metric that describes the number of true positives
in the population of true positives and false negatives combined (e.g. the percentage of POI identified correctly from 
all POI). Precision is a metric that describes the number of true positives in the population of true positives and 
false positives combined (e.g. the percentage of 'real' POI from all people identified as POI).

@@@@@@ 
As described above, I prefer a higher score on recall POI and after that on precision non-POI.
Based on this I would choose the KMeans without PCA. But the project criteria state that precision and recall
should at least be 0.3, therefore the Gaussian Naive Bayes with PCA is the winner here. 
@@@@@@

References:
1) Sklearn Documentation on http://scikit-learn.org
2) Udacity Course documentation
3) Stackoverflow website
4) Python documentation at https://docs.python.org
5) https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_233
6) https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)
7) https://machinelearningmastery.com/how-to-improve-machine-learning-results/


I have cited above the origins of an parts of the submission that were taken from websites,
forums, blog posts, github repositories, etc.
