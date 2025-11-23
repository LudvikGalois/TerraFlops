import math
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from terraflops import TerraFlops
from codecarbon import EmissionsTracker

class TerraScore:
    """Combines ML performance metrics with environmental impact scores."""

    @staticmethod
    def generate_report(y_true, y_pred, tracker: EmissionsTracker, evaluator: TerraFlops):
        """Stop trackers, calculate all metrics, return report dictionary."""
        
        try:
            raw_emissions_kg = tracker.stop()
        except Exception:
            raw_emissions_kg = 0.0
            print("[TerraScore] Warning: CodeCarbon tracker was not running.")

        pue, eff_score = evaluator.stop()
        final_emissions_kg = raw_emissions_kg * pue

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        carbon_per_accuracy = final_emissions_kg / max(accuracy, 0.01)
        carbon_per_f1 = final_emissions_kg / max(macro_f1, 0.01)

        if carbon_per_accuracy > 0:
            log_carbon = math.log10(carbon_per_accuracy)
            carbon_score = max(0, min(10, 10 - ((log_carbon + 8) * 2)))
        else:
            carbon_score = 10.0
        
        composite_score = round((eff_score * 0.7) + (carbon_score * 0.3), 1)

        green_report = {
            "Model_Accuracy": round(accuracy, 4),
            "Model_F1_Macro": round(macro_f1, 4),
            "Precision_Weighted": round(report_dict['weighted avg']['precision'], 4),
            "Recall_Weighted": round(report_dict['weighted avg']['recall'], 4),
            "Raw_IT_Emissions_kg": raw_emissions_kg,
            "PUE": round(pue, 3),
            "Total_Carbon_Footprint_kg": final_emissions_kg,
            "Sustainability_Score": composite_score,
            "Hardware_Efficiency_Score": eff_score,
            "Carbon_Efficiency_Score": round(carbon_score, 1),
            "Carbon_per_Accuracy": round(carbon_per_accuracy, 8),
            "Carbon_per_F1": round(carbon_per_f1, 8)
        }

        return green_report

    @staticmethod
    def compare_models(reports: list):
        """Take list of report dicts, return formatted pandas DataFrame."""
        
        df = pd.DataFrame(reports)
        cols = [
            "Model_Name", "Model_Accuracy", "Sustainability_Score", 
            "Total_Carbon_Footprint_kg", "Carbon_per_Accuracy", "PUE"
        ]
        
        if "Model_Name" not in df.columns:
            df.insert(0, "Model_Name", [f"Model {i+1}" for i in range(len(df))])
            
        existing_cols = [c for c in cols if c in df.columns]
        return df[existing_cols]