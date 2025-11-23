import os
import subprocess
import glob
import time
import sys
import platform
import re
import threading
import json
import math
from typing import Optional, Tuple

DB_FILENAME = "power_data.json"

class HardwareMonitor:
    """Identifies hardware, looks up TDP values, monitors power usage, and estimates RAM power."""

    def __init__(self):
        self.tdp_db = self._load_database()
        self.ram_power_watts = self._estimate_ram_power()

    def _load_database(self) -> dict:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, DB_FILENAME)
            with open(db_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _is_apple_silicon(self) -> bool:
        return sys.platform == 'darwin' and platform.machine() == 'arm64'

    def _estimate_ram_power(self) -> float:
        """Estimate RAM power consumption based on CodeCarbon's methodology. 
        ARM systems use 3W constant, x86 systems use 5W per DIMM with efficiency scaling.
        """
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            print("[TerraFlops] Warning: psutil not installed. Assuming 16GB RAM.")
            total_gb = 16.0

        if self._is_apple_silicon() or platform.machine().lower().startswith('arm'):
            return 3.0

        if total_gb < 16:
            dimms = 2
        elif total_gb <= 64:
            dimms = 4
        elif total_gb <= 256:
            dimms = 8
        else:
            dimms = 16

        total_ram_power = 0.0
        
        for i in range(1, dimms + 1):
            if i <= 4:
                total_ram_power += 5.0
            elif i <= 8:
                total_ram_power += 4.5
            elif i <= 16:
                total_ram_power += 4.0
            else:
                total_ram_power += 3.5

        print(f"[TerraFlops] RAM Estimation: {total_gb:.1f}GB -> ~{dimms} DIMMs ({total_ram_power:.1f}W)")
        return total_ram_power

    def _identify_cpu_name(self) -> str:
        """Detect the CPU model name across different operating systems."""
        model_name = ""
        try:
            if sys.platform == "darwin":
                cmd = ['sysctl', '-n', 'machdep.cpu.brand_string']
                model_name = subprocess.check_output(cmd).decode().strip()
            elif sys.platform == "linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if "model name" in line:
                            model_name = line.split(":")[1].strip()
                            break
            elif sys.platform == "win32":
                cmd = ['wmic', 'cpu', 'get', 'name']
                result = subprocess.check_output(cmd).decode().strip()
                lines = [line.strip() for line in result.split('\n') if line.strip()]
                if len(lines) > 1:
                    model_name = lines[1]
        except Exception:
            pass
        return " ".join(model_name.split())

    def _identify_gpu_name(self) -> str:
        """Detect the GPU model name, prioritizing discrete GPUs over integrated."""
        model_name = ""
        try:
            cmd = ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']
            model_name = subprocess.check_output(cmd).decode().strip()
            return " ".join(model_name.split())
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        platform_name = sys.platform
        
        if platform_name == "win32":
            try:
                cmd = ['wmic', 'path', 'win32_VideoController', 'get', 'name']
                result = subprocess.check_output(cmd).decode().strip()
                lines = [line.strip() for line in result.split('\n') if line.strip()]
                for line in lines[1:]:
                    if "Microsoft" not in line: return line
            except: pass
        elif platform_name == "linux":
            try:
                cmd = ['lspci', '-mm']
                result = subprocess.check_output(cmd).decode().strip()
                for line in result.split('\n'):
                    if '"VGA' in line or '"Display' in line or '"3D' in line:
                        parts = line.split('"')
                        if len(parts) >= 6: return parts[5]
            except: pass
            
        return model_name

    def _is_integrated_gpu(self, gpu_name: str) -> bool:
        """Check if the GPU is integrated based on common naming patterns."""
        igpu_keywords = [
            "uhd graphics", "iris", "intel graphics", "arc graphics",
            "radeon graphics", "radeon tm graphics", 
            "radeon 660m", "radeon 680m", "radeon 760m", "radeon 780m",
            "radeon 880m", "radeon 890m", "vega", "rembrandt"
        ]
        norm = gpu_name.lower()
        for k in igpu_keywords:
            if k in norm: return True
        return False

    def _lookup_tdp(self, device_string: str, category: str) -> float:
        """Search the TDP database for a matching device and return its TDP value."""
        if not device_string or category not in self.tdp_db: return 0.0
        target_str = device_string.lower()
        for vendor in self.tdp_db[category]:
            for key, tdp_val in self.tdp_db[category][vendor].items():
                if key.lower() in target_str: return float(tdp_val)
                key_tokens = key.lower().split()
                if all(token in target_str for token in key_tokens): return float(tdp_val)
        return 0.0

    def get_system_tdp_limit(self) -> float:
        """Calculate total system TDP by summing CPU, GPU, and RAM power limits."""
        total_tdp = 0.0

        cpu_name = self._identify_cpu_name()
        cpu_tdp = self._lookup_tdp(cpu_name, "cpu")
        if cpu_tdp == 0.0:
            cpu_tdp = 30.0 if self._is_apple_silicon() else 65.0
        
        total_tdp += cpu_tdp
        print(f"[TerraFlops] Detected CPU: '{cpu_name}' (Max TDP: {cpu_tdp}W)")

        if not self._is_apple_silicon():
            gpu_name = self._identify_gpu_name()
            if gpu_name:
                gpu_tdp = self._lookup_tdp(gpu_name, "gpu")
                if self._is_integrated_gpu(gpu_name):
                    print(f"[TerraFlops] Detected iGPU: '{gpu_name}'. Included in CPU Package.")
                else:
                    if gpu_tdp == 0.0: gpu_tdp = 250.0
                    total_tdp += gpu_tdp
                    print(f"[TerraFlops] Detected Discrete GPU: '{gpu_name}' (Max TDP: {gpu_tdp}W)")

        total_tdp += self.ram_power_watts

        return total_tdp

    def get_live_power_total(self) -> float:
        """Measure current power consumption from CPU, GPU, and RAM across different platforms."""
        total_watts = 0.0

        if self._is_apple_silicon():
            try:
                cmd = ['sudo', 'powermetrics', '-n', '1', '-i', '100', '--samplers', 'cpu_power']
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0:
                    c = re.search(r'CPU Power:\s+(\d+)\s+mW', res.stdout)
                    g = re.search(r'GPU Power:\s+(\d+)\s+mW', res.stdout)
                    if c: total_watts += float(c.group(1)) / 1000.0
                    if g: total_watts += float(g.group(1)) / 1000.0
                return total_watts + self.ram_power_watts
            except: return 0.0

        try:
            hwmon_paths = glob.glob('/sys/class/hwmon/hwmon*/power1_input')
            for path in hwmon_paths:
                try:
                    with open(path, 'r') as f:
                        mw = float(f.read().strip())
                        if 1.0 < (mw / 1e6) < 1000.0: total_watts += (mw / 1e6)
                except: pass
        except: pass
        
        try:
            res = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                for line in res.stdout.strip().split('\n'):
                    try: total_watts += float(line)
                    except: pass
        except: pass

        try:
            amd_cards = glob.glob('/sys/class/drm/card*/device/hwmon/hwmon*/power1_average')
            for card in amd_cards:
                try:
                    with open(card, 'r') as f:
                        total_watts += float(f.read().strip()) / 1e6
                except: pass
        except: pass

        total_watts += self.ram_power_watts
        
        if total_watts <= self.ram_power_watts and sys.platform == "win32":
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
                cpu_name = self._identify_cpu_name()
                cpu_tdp = self._lookup_tdp(cpu_name, "cpu")
                if cpu_tdp == 0.0:
                    cpu_tdp = 65.0
                total_watts += cpu_tdp * cpu_percent
            except:
                pass

        return total_watts


class TerraFlops:
    """Main TerraFlops class for tracking energy efficiency and calculating sustainability scores."""
    
    CLOUD_PUE_MAP = { "AWS": 1.135, "GCP": 1.10, "AZURE": 1.12, "GENERIC": 1.5 }

    def __init__(self, mode: str = "default", provider: Optional[str] = None):
        self.mode = mode.lower()
        self.provider = provider.upper() if provider else None
        self.hw_monitor = HardwareMonitor() if self.mode == "local_auto" else None
        
        self.monitoring_active = False
        self.monitor_thread = None
        self.system_max_tdp = 0.0
        self.pue_samples = []
        self.sample_interval = 1.0

    def _get_pue_from_load(self, current_watts: float, max_tdp: float) -> float:
        """Calculate Power Usage Effectiveness based on current system load."""
        if max_tdp == 0: return 1.0
        load = current_watts / max_tdp
        
        if load < 0.10: return 1.60
        elif load < 0.30: return 1.40
        elif load < 0.60: return 1.20
        else: return 1.08

    def _monitor_loop(self):
        """Background thread that continuously monitors power usage and calculates PUE."""
        print("[TerraFlops] Background monitoring started...")
        while self.monitoring_active:
            live_watts = self.hw_monitor.get_live_power_total()
            if live_watts > 0 and self.system_max_tdp > 0:
                pue = self._get_pue_from_load(live_watts, self.system_max_tdp)
                self.pue_samples.append(pue)
            time.sleep(self.sample_interval)

    def start(self):
        """Begin monitoring hardware power consumption in the background."""
        if self.mode == "local_auto" and self.hw_monitor and not self.monitoring_active:
            self.system_max_tdp = self.hw_monitor.get_system_tdp_limit()
            
            if self.system_max_tdp == 0:
                print("[TerraFlops] Warning: Could not match hardware. Defaulting to 250W.")
                self.system_max_tdp = 250.0
            
            self.monitoring_active = True
            self.pue_samples = []
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.start()

    def calculate_efficiency_score(self, pue: float) -> float:
        """Convert PUE value to a sustainability score from 1 to 10."""
        if pue <= 1.05: return 10.0
        if pue >= 1.60: return 1.0
        score = 10.0 - ((pue - 1.05) * 16.36)
        return round(max(1.0, min(10.0, score)), 1)

    def stop(self) -> Tuple[float, float]:
        """Stop monitoring and return the final PUE value and efficiency score."""
        final_pue = 1.0
        if self.mode == "cloud":
            final_pue = self.CLOUD_PUE_MAP.get(self.provider, self.CLOUD_PUE_MAP["GENERIC"])
        elif self.mode == "local_auto":
            self.monitoring_active = False
            if self.monitor_thread: self.monitor_thread.join()
            if len(self.pue_samples) > 0:
                final_pue = sum(self.pue_samples) / len(self.pue_samples)
        
        score = self.calculate_efficiency_score(final_pue)
        return final_pue, score