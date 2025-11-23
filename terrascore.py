import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from terraflops import TerraFlops
from codecarbon import EmissionsTracker

class TerraScore:
    """
    Binds Machine Learning Performance Metrics to Environmental Impact Metrics.
    Produces a unified 'Green Report Card' for model comparison.
    """

    @staticmethod
    def generate_report(y_true, y_pred, tracker: EmissionsTracker, evaluator: TerraFlops):
        """
        Stops trackers, calculates metrics, and returns a combined dictionary.
        """
        
        # 1. Stop Environmental Trackers
        # Note: Order matters. Stop codecarbon first to get raw IT emissions.
        try:
            raw_emissions_kg = tracker.stop()
        except Exception:
            raw_emissions_kg = 0.0
            print("[TerraScore] Warning: CodeCarbon tracker was not running.")

        # Stop TerraFlops to get the PUE modifier and Efficiency Score
        # tuple: (pue_val, efficiency_score_0_to_10)
        pue, eff_score = evaluator.stop()

        # 2. Calculate Adjusted Emissions
        # Formula: Raw_IT_Emissions * PUE_Modifier
        final_emissions_kg = raw_emissions_kg * pue

        # 3. Calculate ML Performance Metrics
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # 4. Construct the Unified Report
        green_report = {
            # --- Model Performance ---
            "Model_Accuracy": round(accuracy, 4),
            "Model_F1_Macro": round(macro_f1, 4),
            "Precision_Weighted": round(report_dict['weighted avg']['precision'], 4),
            "Recall_Weighted": round(report_dict['weighted avg']['recall'], 4),
            
            # --- Sustainability Metrics ---
            "Raw_IT_Emissions_kg": raw_emissions_kg,
            "True_PUE": round(pue, 3),
            "Total_Carbon_Footprint_kg": final_emissions_kg,
            
            # --- The "TerraScore" (0-10) ---
            # 10 = Perfect Hardware Utilization (High Load, Low Waste)
            # 0  = Poor Utilization (Idle waste, high overhead)
            "Sustainability_Score": eff_score
        }

        return green_report

    @staticmethod
    def compare_models(reports: list):
        """
        Helper to display a Pandas DataFrame comparing multiple models.
        Input: List of report dictionaries.
        """
        df = pd.DataFrame(reports)
        # Reorder columns for readability
        cols = [
            "Model_Name", "Model_Accuracy", "Sustainability_Score", 
            "Total_Carbon_Footprint_kg", "True_PUE"
        ]
        # Handle case where Model_Name might not be in dict yet
        if "Model_Name" not in df.columns:
            df.insert(0, "Model_Name", [f"Model {i+1}" for i in range(len(df))])
            
        # Return reordered if columns exist, else return raw
        existing_cols = [c for c in cols if c in df.columns]
        return df[existing_cols]