import math
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from terraflops import TerraFlops

class CarbonBenchmark:
    """Tracks compute metrics during training and calculates learning efficiency (k)."""
    
    def __init__(self):
        self.tracker = EmissionsTracker(log_level="error", measure_power_secs=1)
        self.evaluator = TerraFlops(mode="local_auto")
        self.history = []
        self.start_time = None
        self.flop_counter = 0.0
        
    def start_training(self):
        self.tracker.start()
        self.evaluator.start()
        self.start_time = time.time()
        self.history = []
        self.flop_counter = 0.0
        
    def record_epoch(self, epoch: int, accuracy: float, flops_this_epoch: float = 0.0):
        self.flop_counter += flops_this_epoch
        elapsed_time = time.time() - self.start_time
        emissions = self.tracker.flush()
        
        try:
            electricity = self.tracker._total_energy.kwh
        except AttributeError:
            electricity = 0.0 
            
        self.history.append({
            'epoch': epoch + 1,
            'time': elapsed_time,
            'electricity': electricity,
            'co2': emissions,
            'flops': self.flop_counter,
            'accuracy': accuracy
        })
        print(f"[Epoch {epoch+1}] Acc: {accuracy:.4f} | Time: {elapsed_time:.1f}s | CO2: {emissions:.6f}kg")

    def stop_training(self):
        self.tracker.stop()
        self.evaluator.stop()
        
    def plot_learning_curve(self, metric: str = 'co2', task_name: str = "Model"):
        if not self.history:
            return
        ts = [pt[metric] for pt in self.history]
        log_As = [math.log(max(pt['accuracy'], 1e-9)) for pt in self.history]
        
        plt.figure(figsize=(6, 4))
        plt.plot(ts, log_As, marker='o', linestyle='-', color='b', linewidth=2)
        plt.xlabel(f"Compute Measure: {metric.upper()}")
        plt.ylabel("ln(Accuracy)")
        plt.title(f"{task_name}: ln(Accuracy) vs {metric.upper()}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def calculate_k(self, metric: str = 'co2') -> tuple:
        if len(self.history) < 2: return 0.0, 0.0
        ks = []
        for pt1, pt2 in itertools.combinations(self.history, 2):
            t1, A1 = pt1[metric], pt1['accuracy']
            t2, A2 = pt2[metric], pt2['accuracy']
            if A2 == A1: continue
            try:
                term1 = A1 * np.exp(t2, dtype=np.float64)
                term2 = A2 * np.exp(t1, dtype=np.float64)
                val = (term1 - term2) / (A2 - A1)
                if val > 0: ks.append(math.log(val))
            except OverflowError: pass
        if not ks: return 0.0, 0.0
        return np.mean(ks), (np.std(ks, ddof=1) / np.sqrt(len(ks)) if len(ks) > 1 else 0.0)

    def get_metrics_dataframe(self):
        return pd.DataFrame(self.history) if self.history else pd.DataFrame()

    def get_all_k_scores(self):
        metrics = ['time', 'electricity', 'co2', 'flops']
        results = {}
        for m in metrics:
            k_mean, k_sem = self.calculate_k(metric=m)
            results[f"k_{m}"] = k_mean
            results[f"k_{m}_sem"] = k_sem
        return results