
import sys
import os
import joblib
import sklearn
from sklearn.tree import DecisionTreeRegressor

def inspect():
    print(f"Python Version: {sys.version}")
    print(f"Scikit-learn Version: {sklearn.__version__}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'model_v2.joblib')
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    print(f"Loading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        
        # Dig into the pipeline to find a tree
        # Assuming pipeline -> 'regressor' -> 'estimators_' (if Forest)
        
        if hasattr(model, 'named_steps'):
            regressor = model.named_steps.get('regressor') or model.named_steps.get('model')
            if regressor:
                print(f"Regressor type: {type(regressor)}")
                
                # Check for single tree or forest
                tree = None
                if hasattr(regressor, 'estimators_'):
                    tree = regressor.estimators_[0]
                    print("Extracted first tree from Forest.")
                elif isinstance(regressor, DecisionTreeRegressor):
                    tree = regressor
                    print("Regressor is a DecisionTree.")
                
                if tree:
                    print(f"Tree type: {type(tree)}")
                    if hasattr(tree, 'monotonic_cst'):
                        print(f"✅ Attribute 'monotonic_cst' FOUND. Value: {tree.monotonic_cst}")
                    else:
                        print("❌ Attribute 'monotonic_cst' NOT FOUND!")
                else:
                    print("Could not isolate a single DecisionTree from regressor.")
            else:
                 print("Could not find 'regressor' or 'model' step in pipeline.")
        else:
            print("Model is not a Pipeline or has no named_steps.")

    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    inspect()
