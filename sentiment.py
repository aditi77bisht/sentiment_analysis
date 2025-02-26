import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
# This dataset contains text data along with their sentiment labels (positive, negative, or neutral)
df = pd.read_csv("C:/Users/HP/Downloads/sentiment_data_large.csv")


# Step 2: Preprocessing function
# This function converts text to lowercase to maintain uniformity
def preprocess_text(text):
    return text.lower()

# Apply preprocessing to the text column
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Step 3: Convert text data into numerical format using TF-IDF
# TF-IDF (Term Frequency-Inverse Document Frequency) helps in vectorizing text data
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['label']

# Step 4: Splitting Data into Training and Testing Sets
# 80% data is used for training, and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
# Using a simple Naive Bayes classifier which is effective for text classification
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Making Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluating the Model
# Checking how well the model performed using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Visualizing Results
# Plot a confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot sentiment distribution
sns.countplot(x=df['label'], palette='coolwarm')
plt.title("Sentiment Distribution in Dataset")
plt.show()

# Step 9: User Input Processing
# This allows a user to enter a sentence and get its sentiment prediction
user_input = input("Enter a sentence: ")
cleaned_input = preprocess_text(user_input)  # Preprocess the input
input_vectorized = tfidf.transform([cleaned_input]).toarray()  # Convert input text to numerical format
input_prediction = model.predict(input_vectorized)  # Predict sentiment

# Display the result
print("User Input Sentiment:", input_prediction[0])