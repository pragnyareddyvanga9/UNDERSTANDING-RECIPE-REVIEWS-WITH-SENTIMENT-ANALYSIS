# UNDERSTANDING-RECIPE-REVIEWS-WITH-SENTIMENT-ANALYSIS
1. Project Overview 
- This project analyzes user reviews on recipe platforms using Natural Language Processing (NLP) to gain insights into user sentiments, detect fake reviews, and uncover common discussion themes. The pipeline includes sentiment analysis, topic modeling, anomaly detection, and supervised classification.

2. Objectives  
- Classify review sentiment using VADER and TextBlob.  
- Identify main discussion topics using LDA and NMF.  
- Detect fake or abnormal reviews using One-Class SVM.  
- Enhance fake review detection using XGBoost with SMOTE for class balancing.  
- Visualize results to easily interpret patterns and trends.  

3. Tools and Technologies  
- Languages: Python 3  
- Libraries: pandas, numpy, scikit-learn, gensim, NLTK, TextBlob, VADER, matplotlib, seaborn, imbalanced-learn (SMOTE)  
- Platform: Jupyter Notebook  
- Dataset: UCI Machine Learning Repository - Recipe Reviews and User Feedback  

4. Implementation Summary  
i. Data Preprocessing  
   - Removed HTML, punctuation, and stopwords.  
   - Tokenized and lemmatized the text for uniformity.  
ii. Sentiment Analysis  
    - Used both VADER and TextBlob to compute sentiment polarity.  
    - Labeled reviews as Positive, Neutral, or Negative based on combined scores.  
iii. Topic Modeling  
     - Applied LDA (using gensim and sklearn) to detect major themes like taste, prep time, and substitutions.  
     - Also experimented with NMF for comparison.  
iv. Fake Review Detection  
    - Used One-Class SVM to identify unusual or bot-like reviews without needing labeled data.  
    - Evaluated using review text and sentiment features.  
v. XGBoost + SMOTE 
   - Built a labeled dataset using heuristic rules.  
   - Applied SMOTE to handle class imbalance.  
   - Trained XGBoost and achieved 99.1% accuracy, with high precision and recall.  
vi. Visualization  
   - Sentiment results shown using pie charts and bar plots.  
   - Word clouds used for topic modeling visualization.  
   - Scatter plots and heatmaps to explore review properties.

5. Example Input and Output  
Input Example:  
"My family absolutely loved this dish and requested that we have it again"  

Output Example:   
- Sentiment: Positive  
- SVM Output: Normal  
- XGBoost Output: Real Review  

6. Results  
- Sentiment Analysis:  
  Around 92% of reviews were positive, around 5% neutral, and around 3% negative.  
  VADER captured emotional tone well; TextBlob leaned more neutral.
- Topic Modeling:  
  Main themes included quick meals, baking tips, ingredient swaps, and taste/texture feedback.  
- Fake Review Detection (SVM):  
  Flagged around 10% of reviews as suspicious—often vague, emotionally mismatched, or spam-like.  
- XGBoost Classifier:  
  Achieved 99.1% accuracy after SMOTE balancing, with only 48 misclassifications out of 5,448 reviews.  

7. How to Set Up and Run  
- Open the project in Jupyter Notebook  
- Install required packages (if not installed):  
    pip install pandas numpy scikit-learn gensim nltk textblob vaderSentiment matplotlib seaborn imbalanced-learn  
- Download the dataset:  
  [Recipe Reviews and User Feedback (UCI)](https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset)  
- Run the notebook `AIT526DL1_Team 6_Project.ipynb` step by step  

8. How to Re-run / Recompile  
- Ensure all code cells are run in order  
- Re-run preprocessing and models if data changes  
- Visualizations will regenerate as the notebook runs  

9. Lessons Learned  
- Data cleaning is critical — small inconsistencies can affect results.  
- Using multiple sentiment tools gave better confidence in our labels.  
- Combining unsupervised and supervised models provided a deeper understanding of review behavior.  
- Visualizations helped quickly communicate insights.

