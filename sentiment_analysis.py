import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


data = [
    ("I love this product", 1),
    ("This is amazing", 1),
    ("Worst experience ever", 0),
    ("Not good at all", 0),
    ("Totally worth it", 1),
    ("I hate this", 0),
    ("Very disappointing", 0),
    ("Excellent quality!", 1),
    ("I will buy this again", 1),
    ("Terrible. Do not buy", 0),
    ("Pretty good, I'm happy", 1),
    ("It broke after a day", 0),
    ("Exceeded my expectations", 1),
    ("Absolutely awful", 0),
    ("I'm impressed with the performance", 1),
    ("I regret purchasing this", 0),
    ("Five stars", 1),
    ("One star only", 0),
    ("Works as described", 1),
    ("Complete waste of money", 0),
    ("Highly recommended", 1),
    ("Not worth the price", 0),
    ("Fantastic! Loved it", 1),
    ("Disappointed, won't recommend", 0)
]

texts = [t for t, _ in data]
labels = [y for _, y in data]


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    # remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

processed_texts = [preprocess(t) for t in texts]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)
y = labels


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "SVM (linear)": SVC(kernel='linear', probability=True, random_state=42)
}


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {"model": model, "accuracy": acc, "confusion": confusion_matrix(y_test, preds)}
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))


best_name = max(results.keys(), key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.3f})")


def predict_sentiment(text):
    p = preprocess(text)
    v = vectorizer.transform([p])
    pred = best_model.predict(v)[0]
    return "Positive" if pred == 1 else "Negative"


tests = [
    "I hate this",
    "This is wonderful",
    "Not worth the money",
    "Absolutely fantastic product"
]

print("\nSample Predictions:")
for s in tests:
    print(f"'{s}' -> {predict_sentiment(s)}")