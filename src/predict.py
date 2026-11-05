from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "sentiment_model.joblib"

def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python src/train_model.py")
        return

    model = joblib.load(MODEL_PATH)
    print("Sentiment Classifier")
    print("Enter a sentence, or type exit to quit.")

    while True:
        text = input("\nYour text: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            print("Please enter some text.")
            continue

        prediction = model.predict([text])[0]
        print("Predicted sentiment:", prediction)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            print("Class probabilities:")
            for label, probability in zip(model.classes_, probabilities):
                print(f"{label}: {probability:.2%}")

if __name__ == "__main__":
    main()
