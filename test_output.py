import time
import numpy as np
from terraflops import TerraFlops
from codecarbon import EmissionsTracker
from terrascore import TerraScore

# --- Mock Data ---
# Simulating a classification task
y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 100)
# Model A Predictions (Good accuracy)
y_pred_a = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1] * 100) 
# Model B Predictions (Slightly worse accuracy)
y_pred_b = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1] * 100) 

results = []

# ==========================================
# RUN 1: "Model A" (Simulating High Efficiency / Cloud)
# ==========================================
print("\n--- Training Model A (Cloud Mode) ---")

# 1. Init Trackers
# Using 'cloud' mode implies a fixed, efficient PUE (e.g. AWS = 1.135)
evaluator_a = TerraFlops(mode="cloud", provider="AWS")
tracker_a = EmissionsTracker(project_name="Model_A", pue=1.0, save_to_file=False)

# 2. Start
evaluator_a.start()
tracker_a.start()

# 3. Run Workload
time.sleep(2) # Simulating fast, efficient training

# 4. Generate Report
report_a = TerraScore.generate_report(y_test, y_pred_a, tracker_a, evaluator_a)
report_a["Model_Name"] = "Model A (Cloud)"
results.append(report_a)


# ==========================================
# RUN 2: "Model B" (Simulating Inefficient Local Run)
# ==========================================
print("\n--- Training Model B (Local Auto) ---")

# 1. Init Trackers
# 'local_auto' will measure real drivers. 
# If you are running this on a machine with no GPU load, 
# TerraFlops will detect low utilization and give a BAD score.
evaluator_b = TerraFlops(mode="local_auto")
tracker_b = EmissionsTracker(project_name="Model_B", pue=1.0, save_to_file=False)

# 2. Start
evaluator_b.start()
tracker_b.start()

# 3. Run Workload
# We sleep longer to simulate a slower, inefficient model
time.sleep(4) 

# 4. Generate Report
report_b = TerraScore.generate_report(y_test, y_pred_b, tracker_b, evaluator_b)
report_b["Model_Name"] = "Model B (Local)"
results.append(report_b)


# ==========================================
# FINAL COMPARISON
# ==========================================
print("\n\n=== 🏆 FINAL LEADERBOARD ===")
df = TerraScore.compare_models(results)
print(df.to_string(index=False))

print("\nAnalysis:")
print(f"Model A Score: {report_a['Sustainability_Score']}/10 (Cloud is usually efficient)")
print(f"Model B Score: {report_b['Sustainability_Score']}/10 (Local idle/low-load is penalized)")