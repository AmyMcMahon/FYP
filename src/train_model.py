import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
nltk.download('punkt')

# Load your dataset
df = pd.read_csv('../datasets/train/final_labels.csv')
df = df[['level_1', 'body', 'split']]
df = df.dropna(subset=['body'])

df['body'] = df['body'].str.lower()
df['body'] = df['body'].astype(str)  # Convert 'text' column to string data type

# Tokenization and removing stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['tokens'] = df['body'].apply(nltk.word_tokenize)  # Tokenization
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])  # Remove stopwords

# Prepare the features (X) and target (y)
X = df['tokens'].apply(lambda x: ' '.join(x))  # Convert tokens back to string for vectorization
y = df['level_1']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train an SVM model
model = SVC()
model.fit(X_train_vectors, y_train)

# Make predictions
y_pred = model.predict(X_test_vectors)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))
