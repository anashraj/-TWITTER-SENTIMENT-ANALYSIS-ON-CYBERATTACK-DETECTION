# -TWITTER-SENTIMENT-ANALYSIS-ON-CYBERATTACK-DETECTION
This project involves sentiment analysis on a dataset of tweets. The analysis is conducted using Python and various     libraries such as Pandas, Numpy, Scikit-learn, and TextBlob. Below is the chronological order of tasks and the corresponding explanation for each part of the project.

1. Importing Libraries
Various essential libraries for data manipulation, text processing, machine learning, and visualization are imported. These include Pandas for data manipulation, TextBlob for sentiment analysis, Seaborn and Matplotlib for visualization, and Scikit-learn for machine learning tasks.

2. Loading and Inspecting the Data
The dataset is loaded into a Pandas DataFrame and its structure is inspected. This includes checking for missing values and reviewing the column names to understand the dataset's features. Basic data exploration is done using functions like info() and isnull().

3. Data Preprocessing
Text Cleaning: The tweet text undergoes several cleaning steps. The text is converted to lowercase, URLs, mentions, hashtags, and punctuation are removed, and stopwords (common words with little meaningful value) are filtered out. This step ensures that only relevant words are kept for analysis.

Removing Duplicates: After preprocessing, duplicate rows are removed from the dataset to ensure the model trains on unique data.

4. Stemming
Text Stemming: The cleaned text is subjected to stemming using the PorterStemmer. Stemming reduces words to their base form (e.g., "running" becomes "run"), which helps to standardize variations of words and improves the accuracy of the model.

5. Sentiment Analysis
Polarity Calculation: The sentiment of each tweet is evaluated using the TextBlob library, which calculates a polarity score. The polarity score ranges from -1 (negative sentiment) to +1 (positive sentiment).

Classifying Sentiment: Based on the polarity score, tweets are classified into three categories: "Positive", "Neutral", and "Negative". A custom function maps the polarity score to one of these sentiment labels.

6. Sentiment Distribution Visualization
Countplot and Piechart: The distribution of the sentiments (positive, neutral, and negative) is visualized using a countplot and a pie chart. This helps in understanding the overall sentiment trend across the dataset.

7. Wordcloud for Frequent Words in Each Sentiment
Wordcloud Generation: Word clouds are generated for each sentiment category (positive, negative, and neutral). The word clouds highlight the most frequent words in each category, giving a visual representation of the common themes and topics discussed in the tweets.

8. Feature Extraction and Model Training
Vectorization: The text data is transformed into numerical features using a CountVectorizer. This converts the text into a bag-of-words model, where each word or n-gram (word pair) becomes a feature in the dataset.

Splitting Data: The dataset is split into training and testing sets, with 80% of the data used for training the model and 20% reserved for testing. This split allows for evaluating the model's performance on unseen data.

9. Logistic Regression Model
Model Training: A Logistic Regression model is trained on the training data. After training, the model is evaluated on the test data, and various performance metrics such as accuracy, confusion matrix, and classification report are used to assess its effectiveness.

Tuning Hyperparameters with GridSearchCV: GridSearchCV is employed to find the best hyperparameters for the Logistic Regression model, improving its accuracy by testing different values for the regularization parameter C.

10. Support Vector Machine (SVM) Model
Model Training: A Support Vector Machine (SVM) model is also trained and evaluated. Similar to Logistic Regression, SVM is used to classify the tweets based on sentiment, and its performance is assessed using accuracy, confusion matrix, and classification report.

Hyperparameter Tuning: Hyperparameters of the SVM model, such as C, kernel, degree, and gamma, are tuned using GridSearchCV to optimize the model's performance.

11. Model Evaluation and Results
Confusion Matrix: The confusion matrix for both the Logistic Regression and SVM models is plotted to visualize how well the models performed. The matrix shows the true positives, true negatives, false positives, and false negatives, which helps to understand the types of errors the model is making.

12. Model Comparison for Future Analysis
Comparison of Logistic Regression and SVM Models:

Logistic Regression: This model is often a good baseline for binary or multi-class classification problems, especially when the relationships between features and the target are linear. It performed reasonably well in terms of accuracy, but it might struggle with more complex, non-linear patterns in the data.

Support Vector Machine (SVM): SVM performed well on this dataset, especially with the kernel trick, which allows it to handle non-linear relationships between features. It is more complex and computationally expensive than Logistic Regression but might offer better performance for more complicated datasets.

Performance Metrics: Both models were evaluated using accuracy, confusion matrix, and classification report. SVM might provide better performance on some datasets, especially when the data has non-linear boundaries. However, Logistic Regression is faster and easier to interpret, making it suitable for simpler datasets or real-time applications where speed is important.

Recommendation for Future Analysis:

If the dataset is expected to have complex, non-linear relationships, or if the number of features is very large, SVM might be the better choice due to its ability to capture non-linear patterns and higher flexibility in feature space.
If interpretability, speed, and simplicity are more important, Logistic Regression could be preferred. It is particularly useful in situations where model explainability is critical, such as understanding the relationship between specific words in tweets and sentiment.
Considerations for New Data:

For future analysis of new data, it is recommended to try both models and compare their performance using cross-validation to ensure robustness. Since SVM can be more computationally expensive, Logistic Regression can be tested first as a baseline before moving on to more complex models like SVM.
The choice of model should also depend on the size and nature of the incoming data. If the data grows significantly or contains more complex features (e.g., multi-modal data), an SVM or even deep learning methods (e.g., neural networks) might become more suitable.

Final Notes:
Performance Metrics: Accuracy, confusion matrix, and classification report are used to evaluate the models' performance, ensuring that both models are effective for the sentiment analysis task.
Grid Search: Grid search is used to tune the hyperparameters and improve model accuracy, especially for Logistic Regression and SVM models.
Visualization: Word clouds and sentiment distribution graphs provide insights into the content of the tweets and the general sentiment across the dataset.
