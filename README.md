sentiment_analysis
This sentiment analysis project classifies text as positive, negative, or neutral using machine learning. It includes data preprocessing, TF-IDF vectorization, model training, and visualization of results. 


This project focuses on **Sentiment Analysis**, which is a Natural Language Processing (NLP) technique used to determine the sentiment expressed in textual data. The dataset consists of various text samples labeled as **positive, negative, or neutral** sentiments.  

The project begins with **data preprocessing**, where text is cleaned by converting it to lowercase and removing unnecessary characters. The cleaned text is then converted into numerical format using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which helps in transforming words into machine-readable vectors.  

For sentiment classification, we use the **Multinomial Naïve Bayes** algorithm, which is widely used for text classification tasks due to its efficiency. The dataset is split into **training and testing sets** to evaluate model performance. The model is then trained on the training data and tested on the unseen test data.  

To enhance visualization, we include **a confusion matrix** to analyze misclassifications and a **sentiment distribution graph** to understand the overall sentiment trends in the dataset. Additionally, users can **input their own sentences**, and the model predicts the sentiment in real-time.  

This project is useful for applications such as **customer feedback analysis, social media monitoring, and product reviews analysis**, providing insights into public opinion and trends.
