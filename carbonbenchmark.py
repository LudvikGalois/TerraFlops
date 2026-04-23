import math
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from terraflops import TerraFlops

class CarbonBenchmark:
    """Tracks compute metrics during training and calculates learning efficiency (k)."""
    
    def __init__(self):
        # We suppress codecarbon's standard output to keep the notebook clean
        self.tracker = EmissionsTracker(log_level="error", measure_power_secs=1)
        self.evaluator = TerraFlops(mode="local_auto")
        self.history = []
        self.start_time = None
        self.flop_counter = 0.0
        
    def start_training(self):
        """Initializes trackers at the start of training."""
        self.tracker.start()
        self.evaluator.start()
        self.start_time = time.time()
        self.history = []
        self.flop_counter = 0.0
        
    def record_epoch(self, epoch: int, accuracy: float, flops_this_epoch: float = 0.0):
        """Records metrics at the end of each epoch."""
        self.flop_counter += flops_this_epoch
        elapsed_time = time.time() - self.start_time
        
        # Flush the tracker to get up-to-date cumulative energy and emissions
        emissions = self.tracker.flush()
        
        # Depending on CodeCarbon version, total energy is typically stored here
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
        """Stops the hardware trackers safely."""
        self.tracker.stop()
        self.evaluator.stop()
        
    def plot_learning_curve(self, metric: str = 'co2'):
        """Plots log(Accuracy) against the specified compute metric."""
        if not self.history:
            print("No data to plot.")
            return
            
        ts = [pt[metric] for pt in self.history]
        # Calculate log(accuracy). Avoid math domain error if accuracy is exactly 0
        log_As = [math.log(max(pt['accuracy'], 1e-9)) for pt in self.history]
        
        plt.figure(figsize=(8, 5))
        plt.plot(ts, log_As, marker='o', linestyle='-', color='b', linewidth=2)
        plt.xlabel(f"Compute Measure: {metric.upper()}")
        plt.ylabel("ln(Accuracy)")
        plt.title(f"Learning Curve: ln(Accuracy) vs {metric.upper()}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def get_metrics_dataframe(self):
        """Returns the epoch history as a pandas DataFrame."""
        import pandas as pd
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def get_all_k_scores(self):
        """Calculates k_mean and k_sem for all compute metrics."""
        metrics = ['time', 'electricity', 'co2', 'flops']
        results = {}
        for m in metrics:
            k_mean, k_sem = self.calculate_k(metric=m)
            results[f"k_{m}"] = k_mean
            results[f"k_{m}_sem"] = k_sem
        return results
        
    def calculate_k(self, metric: str = 'co2') -> tuple:
        """Calculates the average k and Standard Error in the Mean (SEM) from epoch pairs."""
        if len(self.history) < 2:
            return 0.0, 0.0
            
        ks = []
        for pt1, pt2 in itertools.combinations(self.history, 2):
            t1, A1 = pt1[metric], pt1['accuracy']
            t2, A2 = pt2[metric], pt2['accuracy']
            
            if A2 == A1:
                continue
                
            try:
                # To prevent math overflow when e^(t) gets very large, 
                # we use numpy's float64 exp which handles larger thresholds safely.
                term1 = A1 * np.exp(t2, dtype=np.float64)
                term2 = A2 * np.exp(t1, dtype=np.float64)
                
                val = (term1 - term2) / (A2 - A1)
                
                if val > 0:
                    k = math.log(val)
                    ks.append(k)
            except OverflowError:
                pass
                
        if not ks:
            return 0.0, 0.0
            
        k_mean = np.mean(ks)
        # Calculate standard error in the mean (SEM)
        k_sem = np.std(ks, ddof=1) / np.sqrt(len(ks)) if len(ks) > 1 else 0.0
        
        return k_mean, k_sem