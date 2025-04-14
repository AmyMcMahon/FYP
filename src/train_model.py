import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')

# Load your dataset
df = pd.read_csv('../datasets/train/final_labels.csv')
df = df[['level_1', 'body', 'split']]
df = df.dropna(subset=['body'])

df['body'] = df['body'].str.lower()
df['body'] = df['body'].astype(str)  # Convert 'text' column to string data type

# Tokenization and removing stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))
df['tokens'] = df['body'].apply(nltk.word_tokenize)
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])

# Convert tokens back to string for vectorization
X = df['tokens'].apply(lambda x: ' '.join(x))  
y = df['level_1']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization with N-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use unigrams + bigrams

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_vectors = vectorizer.fit_transform(X_train)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vectors, y_train)

# Train an SVM model with class balancing
svm_model = SVC(kernel='rbf', C=1, class_weight='balanced', gamma='scale')

# Use a pipeline for cleaner training
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('svm', svm_model)
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
