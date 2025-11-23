import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from terraflops import TerraFlops
from codecarbon import EmissionsTracker
from terrascore import TerraScore

print("\n" + "=" * 60)
print("GridSearchCV Benchmark with Sustainability Tracking")
print("=" * 60)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

print(f"\nDataset: Iris ({len(X_train)} train, {len(X_test)} test)")
print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")

results = []

print("\n" + "=" * 60)
print("Training Random Forest with GridSearch")
print("=" * 60)

evaluator_rf = TerraFlops(mode="local_auto")
tracker_rf = EmissionsTracker(project_name="RandomForest_GridSearch", save_to_file=False)

evaluator_rf.start()
tracker_rf.start()

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)

start_time = time.time()
grid_rf.fit(X_train, y_train)
training_time = time.time() - start_time

y_pred_rf = grid_rf.predict(X_test)

report_rf = TerraScore.generate_report(y_test, y_pred_rf, tracker_rf, evaluator_rf)
report_rf["Model_Name"] = "Random Forest"
report_rf["Best_Params"] = str(grid_rf.best_params_)
report_rf["Training_Time_sec"] = round(training_time, 2)
report_rf["Precision"] = round(precision_score(y_test, y_pred_rf, average='weighted'), 4)
report_rf["Recall"] = round(recall_score(y_test, y_pred_rf, average='weighted'), 4)
report_rf["F1_Score"] = round(f1_score(y_test, y_pred_rf, average='weighted'), 4)
results.append(report_rf)

print(f"\nBest params: {grid_rf.best_params_}")
print(f"Training time: {training_time:.2f}s")
print(f"Accuracy: {report_rf['Model_Accuracy']:.4f}")
print(f"Sustainability: {report_rf['Sustainability_Score']}/10")

print("\n" + "=" * 60)
print("Training SVM with GridSearch")
print("=" * 60)

evaluator_svm = TerraFlops(mode="local_auto")
tracker_svm = EmissionsTracker(project_name="SVM_GridSearch", save_to_file=False)

evaluator_svm.start()
tracker_svm.start()

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(random_state=42)
grid_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)

start_time = time.time()
grid_svm.fit(X_train, y_train)
training_time = time.time() - start_time

y_pred_svm = grid_svm.predict(X_test)

report_svm = TerraScore.generate_report(y_test, y_pred_svm, tracker_svm, evaluator_svm)
report_svm["Model_Name"] = "SVM"
report_svm["Best_Params"] = str(grid_svm.best_params_)
report_svm["Training_Time_sec"] = round(training_time, 2)
report_svm["Precision"] = round(precision_score(y_test, y_pred_svm, average='weighted'), 4)
report_svm["Recall"] = round(recall_score(y_test, y_pred_svm, average='weighted'), 4)
report_svm["F1_Score"] = round(f1_score(y_test, y_pred_svm, average='weighted'), 4)
results.append(report_svm)

print(f"\nBest params: {grid_svm.best_params_}")
print(f"Training time: {training_time:.2f}s")
print(f"Accuracy: {report_svm['Model_Accuracy']:.4f}")
print(f"Sustainability: {report_svm['Sustainability_Score']}/10")

print("\n" + "=" * 60)
print("Results")
print("=" * 60)

df = TerraScore.compare_models(results)
print("\n" + df.to_string(index=False))

print("\nDetailed breakdown:")
for result in results:
    print(f"\n{result['Model_Name']}:")
    print(f"  Accuracy:      {result['Model_Accuracy']:.4f}")
    print(f"  Precision:     {result['Precision']:.4f}")
    print(f"  Recall:        {result['Recall']:.4f}")
    print(f"  F1:            {result['F1_Score']:.4f}")
    print(f"  Train time:    {result['Training_Time_sec']}s")
    print(f"  Sustainability: {result['Sustainability_Score']}/10")
    print(f"    - Hardware efficiency:  {result['Hardware_Efficiency_Score']}/10")
    print(f"    - Carbon efficiency:    {result['Carbon_Efficiency_Score']}/10")
    print(f"  Carbon:        {result['Total_Carbon_Footprint_kg']:.6f} kg CO2")
    print(f"  Carbon/Acc:    {result['Carbon_per_Accuracy']:.8f} kg per accuracy point")
    print(f"  PUE:           {result['True_PUE']:.3f}")
    print(f"  Best params:   {result['Best_Params']}")

best_perf = max(results, key=lambda x: x['Model_Accuracy'])['Model_Name']
best_sustain = max(results, key=lambda x: x['Sustainability_Score'])['Model_Name']

print(f"\nBest performance: {best_perf}")
print(f"Best sustainability: {best_sustain}")