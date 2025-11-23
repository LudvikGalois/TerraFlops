# TerraFlops

Score ML models on environmental impact with sustainability ratings from 1-10.

## What it does

TerraFlops gives your models a sustainability score that accounts for both energy consumption and hardware efficiency. Works with CodeCarbon to measure carbon footprint, then adjusts for PUE (Power Usage Effectiveness) to show the actual environmental cost.

Pick your training environment (local hardware or cloud provider) and get a score showing how environmentally bad your model was. Lower power usage doesn't always mean more sustainable - a model that trains quickly but leaves hardware idle wastes energy on cooling and overhead.

## How it works

### local_auto mode (hardware monitoring)

**Hardware detection:**
- Identifies CPU/GPU model names via OS calls (sysctl, /proc/cpuinfo, wmic, lspci, nvidia-smi)
- Looks up max TDP from `power_data.json` (scraped database since some drivers don't expose this)
- Estimates RAM power using CodeCarbon v3 heuristics (3W for ARM, 5W per DIMM for x86)

**Live monitoring:**
- Reads actual power draw from system sensors (hwmon, powermetrics, nvidia-smi)
- Samples power every 1 second during training
- Calculates load percentage = live_power / max_tdp

**PUE calculation:**
- Load < 10%: PUE 1.60 (terrible - mostly idle waste)
- Load < 30%: PUE 1.40 (poor)
- Load < 60%: PUE 1.20 (average)
- Load ≥ 60%: PUE 1.08 (excellent - efficient utilisation)

**Fallbacks:**
- Unknown CPU: 65W default (30W for Apple Silicon)
- Unknown discrete GPU: 250W default
- Windows without sensors: Estimates from CPU percentage × TDP
- Missing hardware: 250W system default

### cloud mode (fixed PUE)

Uses industry-reported PUE values for major cloud providers:
- No hardware detection or live monitoring
- Returns fixed PUE based on provider: AWS 1.135, GCP 1.10, Azure 1.12, Generic 1.5
- Hardware efficiency score is fixed (based on provider PUE)
- Carbon efficiency score still varies based on actual training emissions
- Use when training on cloud VMs where sensor access isn't available

## Quick start

```python
from terraflops import TerraFlops
from codecarbon import EmissionsTracker
from terrascore import TerraScore

# Choose your environment
evaluator = TerraFlops(mode="local_auto")  # or mode="cloud", provider="AWS"
tracker = EmissionsTracker(save_to_file=False, log_level="error")

evaluator.start()
tracker.start()

# Train your model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get sustainability score
report = TerraScore.generate_report(y_test, y_pred, tracker, evaluator)
print(f"Sustainability score: {report['Sustainability_Score']}/10")
print(f"  Hardware efficiency: {report['Hardware_Efficiency_Score']}/10")
print(f"  Carbon efficiency: {report['Carbon_Efficiency_Score']}/10")
print(f"Carbon footprint: {report['Total_Carbon_Footprint_kg']} kg CO2")
print(f"Carbon per accuracy: {report['Carbon_per_Accuracy']} kg")
```

## Understanding the score

**Sustainability Score (1-10):**
- 10/10 = Best possible environmental impact
- 1/10 = Worst environmental impact
- Balances hardware efficiency with absolute carbon cost per work done

**How it's calculated:**

1. **CodeCarbon measures raw energy consumption** during training

2. **TerraFlops calculates PUE** based on hardware utilization (local_auto) or uses fixed cloud values (cloud mode)

3. **Final emissions** = raw emissions × PUE modifier

4. **Two component scores are calculated:**
   - **Hardware Efficiency Score (70% weight):** How well you utilized the machine
     - Based on PUE: `10.0 - ((pue - 1.05) * 16.36)`
     - High load = better score (less waste on idle overhead)
   
   - **Carbon Efficiency Score (30% weight):** Environmental cost per unit of work
     - Based on kg CO2 per accuracy point
     - Lower emissions per performance = better score
     - Uses log scale since values vary by orders of magnitude

5. **Composite Score** = (Hardware Score × 0.7) + (Carbon Score × 0.3)

**Why composite?**
Prevents misleading scores where high-power models get perfect ratings just for maxing out hardware. A 1000W model running at 100% load isn't automatically better than a 200W model at 100% load. The carbon efficiency component rewards models that do the same work with less total environmental impact.

**Real example:**

Random Forest: 4.10s, 0.000019 kg CO2 → **7.1/10**
- Hardware efficiency: 8.6/10 (cloud mode, AWS PUE 1.135)
- Carbon efficiency: 3.5/10 (0.00001878 kg per accuracy point)
- Perfect accuracy but high carbon cost

SVM: 0.05s, 0.000001 kg CO2 → **7.8/10**
- Hardware efficiency: 8.6/10 (cloud mode, AWS PUE 1.135)
- Carbon efficiency: 5.9/10 (0.00000107 kg per accuracy point)
- Slightly lower accuracy but 18x better carbon efficiency

SVM wins on composite score despite 2.2% lower accuracy. Both have identical hardware efficiency in cloud mode (fixed PUE), but SVM's dramatically lower carbon cost per accuracy point gives it the edge. The composite score shows that Random Forest's perfect accuracy came at much higher environmental cost - you can decide if that trade-off's worth it for your use case.

## Choosing your environment

**Training on your own device or bare-metal server:**
```python
evaluator = TerraFlops(mode="local_auto")
```
Monitors actual hardware utilisation to calculate dynamic PUE based on load. Use for non-virtualised environments where you've got direct sensor access - laptops, workstations, physical servers. Gives the most accurate tracking by measuring real hardware load and adjusting PUE accordingly.

**Training on cloud or HPC clusters:**
```python
evaluator = TerraFlops(mode="cloud", provider="AWS")  # or "GCP", "AZURE", "GENERIC"
```
Uses industry-reported PUE values for standardised tracking. Use when:
- Training on cloud VMs (AWS, GCP, Azure) where hardware sensors aren't accessible
- Running on HPC/datacentre infrastructure - use `provider="GENERIC"` for general datacentre average PUE (1.5)
- You want consistent cross-provider comparisons using published datacentre efficiency metrics

**Testing/debugging without hardware monitoring:**
```python
evaluator = TerraFlops(mode="default")
```
Returns fixed PUE of 1.0 (perfect efficiency). Useful for development, testing, or when you only care about raw CodeCarbon emissions without PUE adjustment.

**Limitations:**
- `mode="default"` assumes zero overhead (PUE 1.0), which is unrealistic for actual datacentres. Only use for testing or when you want to ignore infrastructure efficiency entirely.
- Hardware efficiency score will be fixed at 10/10 in default mode since PUE's always optimal.
- Carbon efficiency score still varies based on actual emissions, so you get some sustainability signal even in default mode.
- For accurate sustainability tracking, use `local_auto` (physical devices) or `cloud` (virtualised/datacentre environments) to get realistic PUE values.

## Platform support

**Linux:**
- CPU: /proc/cpuinfo for name, /sys/class/hwmon for power
- GPU: lspci for name, /sys/class/drm for AMD power, nvidia-smi for NVIDIA
- Requires read access to /sys files (or run as sudo)

**macOS:**
- CPU: sysctl for name, powermetrics for power (requires sudo)
- GPU: Included in powermetrics for Apple Silicon
- Intel Macs use system calls

**Windows:**
- CPU: wmic for name, falls back to TDP × usage% for power
- GPU: nvidia-smi for NVIDIA cards, AMD Display Library (ADL) for AMD cards, wmic for integrated
- AMD GPU power via ADL (activity % × TDP), CPU uses estimation fallback

**GPU support:**
- NVIDIA: Direct power via nvidia-smi (all platforms)
- AMD: Direct power via sysfs (Linux) or AMD Display Library/ADL (Windows)
- Intel/integrated: Included in CPU package power (not counted separately)

## Example outputs

### Comparing model types

Run `python test_output.py` to see GridSearchCV comparison on Iris dataset:

```
Random Forest:
  Accuracy:      1.0000
  Train time:    4.10s
  Sustainability: 7.1/10
    - Hardware efficiency:  8.6/10
    - Carbon efficiency:    3.5/10
  Carbon:        0.000019 kg CO2
  Carbon/Acc:    0.00001878 kg per accuracy point
  PUE:           1.135

SVM:
  Accuracy:      0.9778
  Train time:    0.05s
  Sustainability: 7.8/10
    - Hardware efficiency:  8.6/10
    - Carbon efficiency:    5.9/10
  Carbon:        0.000001 kg CO2
  Carbon/Acc:    0.00000107 kg per accuracy point
  PUE:           1.135
```

### Hyperparameter tuning

Run `python hyperparameter_tuning_single_model.py` to see sustainability metrics across 45 Logistic Regression configurations:

```
Top 3 Configurations by Accuracy
==================================================
#1 - C=1.0, solver=saga, max_iter=1000
    Accuracy: 1.0000
    Sustainability: 8.5/10
      - Hardware efficiency: 9.5/10
      - Carbon efficiency: 6.1/10
    Carbon: 0.00000093 kg CO2
    Carbon/Acc: 0.00000093 kg
    Training time: 0.013s

#2 - C=10.0, solver=lbfgs, max_iter=100
    Accuracy: 1.0000
    Sustainability: 8.5/10
      - Hardware efficiency: 9.5/10
      - Carbon efficiency: 6.1/10
    Carbon: 0.00000094 kg CO2
    Carbon/Acc: 0.00000094 kg
    Training time: 0.013s

#3 - C=100.0, solver=newton-cg, max_iter=500
    Accuracy: 1.0000
    Sustainability: 8.4/10
      - Hardware efficiency: 9.5/10
      - Carbon efficiency: 6.0/10
    Carbon: 0.00000095 kg CO2
    Carbon/Acc: 0.00000095 kg
    Training time: 0.015s
```

As a example the above shows the top 3 configs ranked by accuracy with their sustainability metrics. All three achieve perfect accuracy, but with slightly different environmental costs - you can see which hyperparameters give you the best performance whilst also being more sustainable. The goal isn't to sacrifice accuracy for sustainability, but to understand the environmental cost of your performance choices.

## Why this exists

CodeCarbon gives you the raw carbon emissions - TerraFlops adds context about how efficiently you used your hardware to generate those emissions. The sustainability score balances two perspectives:

1. **Hardware efficiency:** Did you use the machine well or waste energy on idle overhead?
2. **Carbon efficiency:** What was the total environmental cost per unit of work done?

A model can have low total emissions but still be wasteful if it left hardware idle (poor utilisation = high cooling/overhead waste). Conversely, a model can max out hardware efficiently but emit massive carbon if it's training on power-hungry equipment.

The composite score gives you both angles so you can make informed choices  whether that's maximising the hardware you've got, minimising absolute environmental impact, or finding the best balance.

## Contributing

Missing hardware in `power_data.json`? Open an issue with your CPU/GPU model name and TDP value from the manufacturer's spec sheet.
