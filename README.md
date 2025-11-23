# TerraFlops

Score ML models on environmental impact with sustainability ratings from 1-10.

## What it does

TerraFlops gives your models a sustainability score that accounts for both energy consumption and hardware efficiency. Works with CodeCarbon to measure carbon footprint, then adjusts for PUE (Power Usage Effectiveness) to show the real environmental cost.

Choose your training environment (local hardware or cloud provider) and get a score showing how bad your model was for the environment. Lower power usage doesn't always mean more sustainable - a model that trains fast but leaves hardware idle wastes energy on cooling and overhead.

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
- Load ≥ 60%: PUE 1.08 (excellent - efficient utilization)

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
tracker = EmissionsTracker(save_to_file=False)

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
Prevents misleading scores where high-power models get perfect ratings just for maxing out hardware. A 1000W model running at 100% load isn't automatically better than a 200W model at 100% load. The carbon efficiency component rewards models that accomplish the same work with less total environmental impact.

**Real example:**

Random Forest: 34.6W avg, 4.23s, 0.000019 kg CO2 → **7.5/10**
- Hardware efficiency: 9.2/10 (high utilization, PUE 1.10)
- Carbon efficiency: 3.5/10 (0.000019 kg per accuracy point)
- Used more total power but efficiently utilized hardware

SVM: 5.4W avg, 0.05s, 0.000001 kg CO2 → **7.0/10**
- Hardware efficiency: 7.5/10 (lower utilization, PUE 1.20)
- Carbon efficiency: 5.9/10 (0.000001 kg per accuracy point)
- Used less total power with better carbon cost per performance

Random Forest edges ahead due to better hardware utilization, but the gap is smaller than pure PUE scoring would suggest. The composite score recognizes that while Random Forest used the machine well, it still emitted 18x more carbon per accuracy point. Users can see both perspectives and make informed choices based on their priorities.

## Choosing your environment

**Training on your own hardware:**
```python
evaluator = TerraFlops(mode="local_auto")
```
Monitors actual hardware utilization to calculate dynamic PUE.

**Training on cloud (AWS, GCP, Azure):**
```python
evaluator = TerraFlops(mode="cloud", provider="AWS")  # or "GCP", "AZURE", "GENERIC"
```
Uses industry-reported PUE values for major cloud providers.

**Testing/debugging:**
```python
evaluator = TerraFlops(mode="default")
```
Skips monitoring, returns PUE 1.0.

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
- GPU: nvidia-smi for NVIDIA cards, wmic for integrated
- No native power reading for CPU/AMD GPUs (uses estimation)

**GPU support:**
- NVIDIA: Direct power via nvidia-smi (all platforms)
- AMD: Direct power via sysfs (Linux only)
- Intel/integrated: Included in CPU package power (not counted separately)

## Example output

Run `python test_output.py` to see GridSearchCV comparison on Iris dataset:

```
Random Forest:
  Accuracy:      1.0000
  Train time:    4.23s
  Sustainability: 7.5/10
    - Hardware efficiency:  9.2/10
    - Carbon efficiency:    3.5/10
  Carbon:        0.000019 kg CO2
  Carbon/Acc:    0.00001879 kg per accuracy point
  PUE:           1.100

SVM:
  Accuracy:      0.9778
  Train time:    0.05s
  Sustainability: 7.0/10
    - Hardware efficiency:  7.5/10
    - Carbon efficiency:    5.9/10
  Carbon:        0.000001 kg CO2
  Carbon/Acc:    0.00000113 kg per accuracy point
  PUE:           1.200
```

Random Forest edges ahead on composite score (7.5 vs 7.0) with higher hardware utilization, while SVM is more carbon-efficient per accuracy point. The breakdown lets you see both hardware efficiency and absolute environmental cost, helping you choose based on your priorities.

## Why this exists

CodeCarbon measures energy consumption but misses the efficiency context. TerraFlops adds a sustainability score that balances two perspectives:

1. **Hardware efficiency:** Did you use the machine well or waste energy on idle overhead?
2. **Carbon efficiency:** What was the total environmental cost per unit of work done?

A model can have low total emissions but still be wasteful if it left hardware idle (poor utilization = high cooling/overhead waste). Conversely, a model can max out hardware efficiently but emit massive carbon if it's training on power-hungry equipment.

The composite score gives you both angles so you can pick models that match your priorities - whether that's maximizing the hardware you have, minimizing absolute environmental impact, or finding the best balance.

## Contributing

Missing hardware in `power_data.json`? Open an issue with your CPU/GPU model name and TDP value from the manufacturer's spec sheet.
