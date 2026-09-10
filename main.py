from src.utils import load_dataset
from src.preprocessing import clean_text
from src.feature_extraction import create_features
from src.train_model import train_classifier
from src.evaluation import evaluate_model
from src.prediction import predict_message


# 1. Load dataset
data = load_dataset("data/SMSSpamCollection")

# 2. Clean messages
data["cleaned_message"] = data["message"].apply(clean_text)

# 3. Convert text into numerical features
features, vectorizer = create_features(data["cleaned_message"])

# 4. Train the ML classifier
model, X_test, y_test = train_classifier(features, data["label"])

# 5. Evaluate the model
accuracy, report = evaluate_model(model, X_test, y_test)

print("\n===== AI SPAM MESSAGE DETECTOR =====")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

# 6. Test with a new message
message = input("\nEnter a message to check: ")

prediction, probability = predict_message(
    message,
    model,
    vectorizer,
    clean_text
)

print("\n" + "=" * 40)
print("       SPAM DETECTION RESULT")
print("=" * 40)
print(f"Message     : {message}")
print(f"Prediction  : {prediction.upper()}")
print(f"Confidence  : {probability * 100:.2f}%")
print("=" * 40)