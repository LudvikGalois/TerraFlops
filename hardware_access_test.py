import time
import sys
import os
import json

# Add current directory to path
sys.path.append(os.getcwd())

print("--- TerraFlops Diagnostic Tool (v2.3) ---\n")

# 1. Check for Dependencies
try:
    from terraflops import HardwareMonitor
    print("✅ Library: 'terraflops.py' found.")
except ImportError:
    print("❌ Error: Could not import 'terraflops.py'.")
    sys.exit(1)

# 2. Check for Database (Relative to script)
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "power_data.json")

if os.path.exists(db_path):
    try:
        with open(db_path, 'r') as f:
            json.load(f)
        print(f"✅ Database: Found at {db_path}")
    except json.JSONDecodeError:
        print("❌ Error: 'power_data.json' is malformed JSON.")
else:
    print(f"❌ Error: 'power_data.json' not found at {db_path}")

def run_diagnostics():
    print("\n[Initializing Hardware Monitor...]")
    monitor = HardwareMonitor()
    
    # --- CPU CHECKS ---
    print(f"\n[CPU Diagnostics]")
    cpu_name = monitor._identify_cpu_name()
    print(f"   > Detected String: '{cpu_name}'")
    
    cpu_tdp = monitor._lookup_tdp(cpu_name, "cpu")
    if cpu_tdp > 0:
        print(f"   > Database Match:  ✅ Found (Max TDP: {cpu_tdp} W)")
    else:
        print(f"   > Database Match:  ❌ Not Found (Defaulting fallback)")
        print(f"     (Tip: Add '{cpu_name}' to power_data.json if it's missing)")

    # --- GPU CHECKS ---
    print(f"\n[GPU Diagnostics]")
    if monitor._is_apple_silicon():
        print("   > Platform: Apple Silicon (GPU integrated in SoC)")
        gpu_tdp = 0.0
    else:
        gpu_name = monitor._identify_gpu_name()
        print(f"   > Detected String: '{gpu_name}'")
        
        if gpu_name:
            # Check if it is an Integrated GPU first
            if monitor._is_integrated_gpu(gpu_name):
                print(f"   > Type:            ℹ️  Integrated GPU (iGPU)")
                print(f"   > Status:          ✅ Correctly Ignored (Power bundled with CPU)")
                gpu_tdp = 0.0
            else:
                # Discrete GPU Logic
                print(f"   > Type:            Discrete GPU (dGPU)")
                gpu_tdp = monitor._lookup_tdp(gpu_name, "gpu")
                if gpu_tdp > 0:
                    print(f"   > Database Match:  ✅ Found (Max TDP: {gpu_tdp} W)")
                else:
                    print(f"   > Database Match:  ❌ Not Found (Defaulting fallback)")
        else:
            print("   > Status: No dedicated GPU detected via drivers.")
            gpu_tdp = 0.0

    # --- TOTAL SYSTEM LIMIT ---
    print(f"\n[System Limit Calculation]")
    # We capture the print output from get_system_tdp_limit to see the logic flow
    total_limit = monitor.get_system_tdp_limit()
    print(f"   > Final Max Power Limit: {total_limit:.2f} W")

    # --- LIVE SENSOR LOOP ---
    print(f"\n[Live Power Monitor]")
    print("Reading sensors for 5 seconds (Ctrl+C to stop)...")
    print("-" * 65)
    print(f"{'Time':<8} | {'Live Power':<12} | {'Max TDP':<10} | {'Utilization %':<15}")
    print("-" * 65)

    try:
        for i in range(5):
            live_watts = monitor.get_live_power_total()
            
            # Avoid division by zero for visualization
            util_percent = 0.0
            if total_limit > 0:
                util_percent = (live_watts / total_limit) * 100.0
            
            print(f"{i+1:<8} | {live_watts:<8.2f} W   | {total_limit:<8.2f} W | {util_percent:<6.1f} %")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("-" * 65)
    
    # Final Verdict
    if live_watts == 0.0:
        print("\n❌ DIAGNOSTIC FAILURE: Power sensors returned 0.0 W.")
        print("   - Apple: Needs 'sudo'.")
        print("   - Linux: Needs 'sudo' or permissions on /sys/class/powercap.")
        print("   - Windows: Drivers not currently supported.")
    elif total_limit == 250.0 and "250" not in str(cpu_tdp):
        print("\n⚠️  WARNING: System is running on Fallback TDP (250W).")
        print("   The database did not find your specific hardware.")
    else:
        print("\n✅ SUCCESS: TerraFlops is ready to run.")

if __name__ == "__main__":
    run_diagnostics()