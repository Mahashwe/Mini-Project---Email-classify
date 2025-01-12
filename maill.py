import pandas as pd  # pandas to handle raw data

# Load dataset
df = pd.read_csv("C:\\Users\\smaha\\Mini Projects\\Email Classification\\email.csv")

# Check for missing values (optional)
# print(df.isnull().sum())

from sklearn.feature_extraction.text import CountVectorizer  # BOW approach to convert text to vectors

# Create the vectorizer instance
vectorizer = CountVectorizer()

# Convert the messages into vectors
x = vectorizer.fit_transform(df['Message']).toarray()  # 'Message' column contains email text
y = df['Category']  # 'Category' column contains labels (ham/spam)

from sklearn.model_selection import train_test_split  # Split data into train and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression  # Logistic Regression model

# Initialize and train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score  # Accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Take user input

user_email = input("Enter your email to be classified:")

# Vectorize the user's input email using the same vectorizer
user_vectorized = vectorizer.transform([user_email]).toarray()  # Use transform(), not fit_transform()

# Predict the category (ham/spam) for the user input email
user_prediction = model.predict(user_vectorized)

# Output the prediction
print('Predictions:', user_prediction)
