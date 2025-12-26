import pickle
import os
import sys

def verify():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'model_v2.joblib')
    
    print(f"Checking model at: {model_path}")
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    import joblib
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        print("Pipeline steps:", [step[0] for step in model.steps])
    except Exception as e:
        print(f"FAILED to load model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
