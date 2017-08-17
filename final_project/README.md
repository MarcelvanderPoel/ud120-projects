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
features, from which from_poi_perc had the hightest score. 


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

I looked at the following classification algorithms:
1) Gaussian Naive Bayes
2) Decision Tree
3) SVM

What do I find more important when judging the algorithm, precision, recall or F1?
Would I rather not overlook any poi, or would I rather prevent selecting/accusing the wrong poi and why? 

## Tuning an algorithm

## Validation

## Evaluation Metrics

