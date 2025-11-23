# TerraFlops

Track the real energy efficiency of your ML training runs, not just raw power consumption.

## What it does

TerraFlops wraps CodeCarbon to calculate your actual PUE (Power Usage Effectiveness) during model training. Instead of just measuring watts, it tells you how efficiently you're using those watts.

A model that uses 100W at high load can be more sustainable than one using 30W mostly idle.

## Quick start

```python
from terraflops import TerraFlops
from codecarbon import EmissionsTracker
from terrascore import TerraScore

# Start tracking
evaluator = TerraFlops(mode="local_auto")
tracker = EmissionsTracker(save_to_file=False)

evaluator.start()
tracker.start()

# Train your model here
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get results
report = TerraScore.generate_report(y_test, y_pred, tracker, evaluator)
print(f"Sustainability score: {report['Sustainability_Score']}/10")
print(f"True PUE: {report['True_PUE']}")
```

## Understanding True PUE

PUE = Total System Power / IT Equipment Power

**Lower is better.**

- **1.0** = Perfect efficiency (impossible in practice)
- **1.08-1.15** = Excellent (Google/AWS datacenters, or your laptop under full load)
- **1.2-1.4** = Average (typical office datacenter, or laptop at moderate load)
- **1.5+** = Poor (old datacenters, or your laptop sitting mostly idle)

### Why PUE matters more than raw power

Example from the test output:

- **Random Forest**: 34.5W average, 4.08s training → PUE 1.12 → Score 8.9/10
- **SVM**: 5.4W average, 0.05s training → PUE 1.20 → Score 7.5/10

Random Forest gets a better sustainability score despite using 6x more power because it actually utilized the hardware. Running your CPU at 10% load wastes the idle power on cooling and overhead.

## Modes

### local_auto (recommended)

Monitors your actual hardware in real-time:
- Detects CPU/GPU via system calls
- Looks up TDP values from built-in database
- Tracks live power draw through OS sensors
- Calculates PUE based on utilization

Use this when training on your own hardware.

### cloud

Uses fixed PUE values for cloud providers:
- AWS: 1.135
- GCP: 1.10
- Azure: 1.12
- Generic: 1.5

```python
evaluator = TerraFlops(mode="cloud", provider="AWS")
```

Use this when training on cloud instances where you can't read hardware sensors.

## Sustainability score

Converts PUE to a 1-10 scale:
- **10/10**: PUE ≤ 1.05
- **8-9/10**: PUE 1.05-1.15 (good)
- **5-7/10**: PUE 1.15-1.40 (average)
- **1-4/10**: PUE 1.40-1.60 (poor)
- **1/10**: PUE ≥ 1.60 (terrible)

## Platform support

- **Linux**: Full hardware detection via sysfs/hwmon
- **macOS**: Apple Silicon via powermetrics, Intel via system calls
- **Windows**: CPU via wmic, GPU via nvidia-smi, fallback to TDP estimation

GPU support:
- NVIDIA: Direct power readings via nvidia-smi
- AMD: Power readings via sysfs on Linux
- Intel/integrated: Included in CPU package power

## Example: GridSearchCV benchmark

See `test_output.py` for a complete example comparing Random Forest vs SVM with hyperparameter tuning on the Iris dataset.

Output includes:
- Model accuracy, precision, recall, F1
- Training time
- Sustainability score
- Carbon footprint
- True PUE
- Best hyperparameters

```bash
python test_output.py
```

## Why this exists

CodeCarbon measures energy consumption but doesn't tell you if you're being efficient. A model might use less total energy but still be wasteful if it leaves hardware idle. TerraFlops adds the context you need to make informed decisions about model selection and infrastructure.
