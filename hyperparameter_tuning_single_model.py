import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from terraflops import TerraFlops
from codecarbon import EmissionsTracker
from terrascore import TerraScore

print("\nLogistic Regression Hyperparameter Tuning")
print("=" * 50)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga', 'newton-cg'],
    'max_iter': [100, 500, 1000]
}

param_combinations = [
    {'C': c, 'penalty': p, 'solver': s, 'max_iter': m}
    for c in param_grid['C']
    for p in param_grid['penalty']
    for s in param_grid['solver']
    for m in param_grid['max_iter']
]

print(f"Testing {len(param_combinations)} configs...\n")

results = []

for idx, params in enumerate(param_combinations, 1):
    print(f"Config {idx}/{len(param_combinations)}...", end=" ")
    
    evaluator = TerraFlops(mode="local_auto")
    tracker = EmissionsTracker(
        project_name=f"LogReg_Config_{idx}",
        save_to_file=False,
        log_level="error"
    )
    
    evaluator.start()
    tracker.start()
    
    model = LogisticRegression(**params, random_state=42)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    report = TerraScore.generate_report(y_test, y_pred, tracker, evaluator)
    
    report["Config_ID"] = idx
    report["C"] = params['C']
    report["Solver"] = params['solver']
    report["Max_Iter"] = params['max_iter']
    report["Training_Time_sec"] = round(training_time, 3)
    
    results.append(report)
    
    print(f"done")

print("\n" + "=" * 50)
print("Top 3 Configurations by Accuracy")
print("=" * 50)

df = pd.DataFrame(results)
df = df.sort_values('Model_Accuracy', ascending=False)

for rank, (idx, row) in enumerate(df.head(3).iterrows(), 1):
    print(f"\n#{rank} - C={row['C']}, solver={row['Solver']}, max_iter={row['Max_Iter']}")
    print(f"    Accuracy: {row['Model_Accuracy']:.4f}")
    print(f"    Sustainability: {row['Sustainability_Score']}/10")
    print(f"      - Hardware efficiency: {row['Hardware_Efficiency_Score']:.1f}/10")
    print(f"      - Carbon efficiency: {row['Carbon_Efficiency_Score']:.1f}/10")
    print(f"    Carbon: {row['Total_Carbon_Footprint_kg']:.8f} kg CO2")
    print(f"    Carbon/Acc: {row['Carbon_per_Accuracy']:.8f} kg")
    print(f"    Training time: {row['Training_Time_sec']:.3f}s")
