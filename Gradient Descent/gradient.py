from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample email dataset
emails = [
    "Win money now",
    "Meeting scheduled tomorrow",
    "Claim your free prize",
    "Project report is ready",
    "Congratulations you won cash",
    "Please attend the lecture",
    "Free offer click now",
    "Submit your assignment",
    "You have won a lottery",
    "Let us meet for lunch",
    "Get free gift today",
    "Your class starts at ten",
    "Urgent claim your reward",
    "Please send the meeting notes",
    "Earn cash from home",
    "The exam timetable is released",
    "Click here to win a prize",
    "Your appointment is confirmed",
    "Exclusive offer just for you",
    "Please review the project"
]

# 0 = Not Spam, 1 = Spam
labels = [
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0
]

# Split data into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    emails, labels, test_size=0.30, random_state=42, stratify=labels
)

# Convert email text into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Define classifiers
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
    "SVM": LinearSVC(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    ))

# Predict a new email
new_email = ["Congratulations! Claim your free cash prize now"]
new_features = vectorizer.transform(new_email)

print("\nNew Email:", new_email[0])
for name, model in models.items():
    prediction = model.predict(new_features)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    print(name, "Prediction:", result)