#!/usr/bin/env python3
"""CSI Dashboard with Fall Detection
Based on research: Ensemble Framework for Fall Detection Using Multivariate Wi-Fi CSI

Key features:
- Butterworth low-pass filtering (0.1-10 Hz)
- Sharp spike detection (fall signature)
- Quick stabilization detection (post-fall stillness)
- Combined time + frequency domain features
"""

import time, asyncio, logging, queue
from pathlib import Path
from collections import deque
import numpy as np
import serial
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import uvicorn
import sqlite3
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.stats import skew, kurtosis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csi_light")

CONFIG = {
    "serial_port": "/dev/ttyACM0",
    "baud_rate": 115200,
    "smoothing_window": 50,
    "ema_alpha": 0.1,
    "presence_variance_threshold": 0.0005,
    "movement_threshold": 1.0,
    
    # Fall Detection Parameters (from research)
    "fall_window_ms": 500,        # Short window for fall detection
    "sample_rate": 50,           # Estimated CSI sample rate
    "fall_spike_threshold": 2.0,    # Amplitude spike threshold
    "fall_still_threshold": 0.05,    # Stillness after spike = fall
    "fall_confirm_window_ms": 1500,  # 1.5s to confirm fall
    "fall_spike_to_still_ms": 800,   # Must go still within 800ms after spike
    
    # Risk classification
    "risk_ema_alpha": 0.3,
    
    "calibration_samples": 200,
    "spatial_bins": 8,
    "db_path": "data/light_csi.db",
}


class CSIReader:
    def __init__(self, port, baud, output_queue):
        self.port = port
        self.baud = baud
        self.output_queue = output_queue
        self.running = True
        self._buffer = b""
        
    def run(self):
        try:
            ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=0.5)
            ser.reset_input_buffer()
            time.sleep(0.5)
            
            while self.running:
                try:
                    data = ser.read(2048)
                    if data:
                        self._buffer += data
                        self._extract_csi_lines()
                except:
                    pass
                time.sleep(0.01)
            ser.close()
        except Exception as e:
            logger.error(f"Serial reader error: {e}")
    
    def _extract_csi_lines(self):
        text = self._buffer.decode("utf-8", errors="replace")
        while "CSI_DATA," in text:
            start = text.find("CSI_DATA,")
            end = -1
            search_start = start + 10
            for _ in range(100):
                bracket_pos = text.find("]", search_start)
                if bracket_pos == -1:
                    break
                remaining = text[bracket_pos+1:bracket_pos+10]
                if "\n" in remaining or remaining.startswith("CSI_DATA,") or remaining.startswith("I ("):
                    end = bracket_pos + 1
                    break
                search_start = bracket_pos + 1
            
            if end == -1:
                if start > 0:
                    self._buffer = text[start].encode()
                break
            
            line = text[start:end].strip()
            if line:
                self.output_queue.put(line)
            text = text[end:]
            self._buffer = text.encode()


class CSIProcessor:
    def __init__(self, config):
        self.config = config
        self.smoothing_window = config["smoothing_window"]
        self.ema_alpha = config["ema_alpha"]
        self.movement_threshold = config["movement_threshold"]
        self.presence_variance_threshold = config["presence_variance_threshold"]
        self.spatial_bins = config["spatial_bins"]
        
        # Signal buffers
        self.amplitude_history = deque(maxlen=self.smoothing_window * 2)
        self.phase_history = deque(maxlen=self.smoothing_window)
        self.rssi_history = deque(maxlen=100)
        self.presence_value_history = deque(maxlen=100)
        
        # Processed amplitude buffer for fall detection
        self.fall_buffer = deque(maxlen=int(config["fall_window_ms"] * config["sample_rate"] / 1000 * 2))
        
        # EMA smoothing
        self.smoothed_presence = 0.0
        self.smoothed_movement = 0.0
        self.smoothed_rssi = -50.0
        
        # Fall detection state
        self.spike_detected = False
        self.spike_time = None
        self.spike_amplitude = None
        self.last_amplitude = None
        self.fall_risk = 0.0
        self.fall_confirmed = False
        
        # Risk classification
        self.risk_history = deque(maxlen=50)
        self.current_movement_type = "NORMAL"
        self.risk_level = 0.0
        
        # Calibration
        self.calibration_buffer = deque(maxlen=config["calibration_samples"])
        self.is_calibrated = False
        self.baseline_amplitude = None
        self.position_history = deque(maxlen=50)
        self.frame_count = 0
        self.calibration_in_progress = False
        
        # Stored values for /status API
        self.last_movement_type = "NORMAL"
        self.last_risk_level = 0.0
        
        # Initialize Butterworth filter
        self.filter_coeffs = None
        self._init_filter()
        
    def _init_filter(self):
        """Initialize Butterworth low-pass filter (0.1-10 Hz for human movement)"""
        try:
            low = self.config.get("bandpass_low", 0.1) / (self.config["sample_rate"] / 2)
            high = min(self.config.get("bandpass_high", 10.0) / (self.config["sample_rate"] / 2), 0.99)
            if low >= high:
                low = 0.001
            b, a = butter(4, [low, high], btype='band')
            self.filter_coeffs = (b, a)
        except Exception as e:
            logger.warning(f"Filter init failed: {e}")
            self.filter_coeffs = None
    
    def _apply_filter(self, signal):
        """Apply Butterworth bandpass filter"""
        if self.filter_coeffs is None or len(signal) < 10:
            return signal
        try:
            return filtfilt(self.filter_coeffs[0], self.filter_coeffs[1], signal)
        except:
            return signal
    
    def start_calibration(self):
        self.calibration_buffer.clear()
        self.presence_value_history.clear()
        self.is_calibrated = False
        self.baseline_amplitude = None
        self.calibration_in_progress = True
        self.smoothed_presence = 0.0
        self.smoothed_movement = 0.0
        logger.info("Calibration started")
    
    def _detect_fall(self, amplitude_buffer):
        """
        Fall detection based on research paper approach:
        1. Sharp amplitude spike (impact)
        2. Quick stabilization (person fallen)
        """
        if len(amplitude_buffer) < 20:
            return 0.0, False
        
        signal = np.array(list(amplitude_buffer))
        
        # Apply Butterworth filter for human movement range
        signal = self._apply_filter(signal)
        
        now = time.time()
        
        # Calculate current metrics
        amp_mean = np.mean(signal)
        amp_std = np.std(signal)
        amp_range = np.max(signal) - np.min(signal)
        
        # Time-domain features (from research)
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        
        # Velocity change (first derivative)
        velocity = np.diff(signal)
        velocity_change = np.mean(np.abs(velocity))
        
        # Peak-to-peak ratio
        peak_to_peak = amp_range / (amp_mean + 1e-10)
        
        # Skewness and Kurtosis
        signal_skew = skew(signal)
        signal_kurt = kurtosis(signal)
        
        # === FALL DETECTION LOGIC ===
        # Fall signature: sharp spike -> sudden stillness
        
        # Check for sharp amplitude change
        if self.last_amplitude is not None:
            amplitude_change = abs(amp_mean - self.last_amplitude) / (self.last_amplitude + 1e-10)
            
            # SPIKE DETECTED: Large sudden amplitude change
            if amplitude_change > self.config["fall_spike_threshold"] and not self.spike_detected:
                self.spike_detected = True
                self.spike_time = now
                self.spike_amplitude = amp_mean
                logger.info(f"FALL SPIKE DETECTED: change={amplitude_change:.2f}")
            
            # After spike, check for quick stabilization
            if self.spike_detected and self.spike_time:
                time_since_spike = now - self.spike_time
                
                # Must go still within confirm window
                if time_since_spike < self.config["fall_spike_to_still_ms"] / 1000:
                    # Check if amplitude has stabilized (low variance = person fallen)
                    if amp_std < self.config["fall_still_threshold"]:
                        logger.warning("FALL CONFIRMED!")
                        self.fall_confirmed = True
                        self.spike_detected = False
                        return 1.0, True
                
                # Reset if too much time passed
                if time_since_spike > self.config["fall_confirm_window_ms"] / 1000:
                    self.spike_detected = False
                    self.spike_time = None
        
        self.last_amplitude = amp_mean
        
        # Calculate fall risk score based on features
        fall_risk = 0.0
        
        # Factor 1: High peak-to-peak ratio (unusual movement)
        if peak_to_peak > 2.0:
            fall_risk += 0.3
        elif peak_to_peak > 1.5:
            fall_risk += 0.15
        
        # Factor 2: High velocity change (sudden movement)
        if velocity_change > 0.5:
            fall_risk += 0.25
        elif velocity_change > 0.3:
            fall_risk += 0.1
        
        # Factor 3: Abnormal skewness (asymmetric distribution = unusual)
        if abs(signal_skew) > 2.0:
            fall_risk += 0.2
        elif abs(signal_skew) > 1.5:
            fall_risk += 0.1
        
        # Factor 4: Low kurtosis (sudden spikes flatten distribution)
        if signal_kurt < -0.5:
            fall_risk += 0.15
        
        # Factor 5: Low MAD (very stable = person on ground)
        if mad < 0.05 and self.last_amplitude is not None:
            fall_risk += 0.1
        
        return min(1.0, fall_risk), False
    
    def _classify_movement_type(self, fall_risk, movement_detected):
        """
        Classify movement as:
        - FALL: Immediate danger
        - HIGH_RISK: Unusual movement pattern
        - NORMAL: Regular walking/movement
        """
        if not movement_detected:
            return "NORMAL", 0.0
        
        if self.fall_confirmed:
            self.fall_confirmed = False  # Reset after reporting
            return "FALL", 1.0
        
        # Calculate combined risk
        risk = fall_risk * 0.7 + (self.fall_risk * 0.3)
        risk = min(1.0, risk)
        
        # EMA smoothing
        self.fall_risk = self.config["risk_ema_alpha"] * risk + (1 - self.config["risk_ema_alpha"]) * self.fall_risk
        self.risk_history.append(self.fall_risk)
        
        # Classify
        if self.fall_risk > 0.7:
            return "HIGH_RISK", self.fall_risk
        else:
            return "NORMAL", self.fall_risk
    
    def process(self, line):
        if not line or not line.startswith("CSI_DATA,"):
            return None
        
        try:
            bracket_start = line.find("\"[")
            bracket_end = line.find("]", bracket_start)
            if bracket_start == -1 or bracket_end == -1:
                return None
            
            csi_raw = line[bracket_start+2:bracket_end]
            csi_values = []
            for v in csi_raw.split(","):
                v = v.strip()
                try:
                    csi_values.append(int(v))
                except ValueError:
                    continue
            
            if len(csi_values) < 4:
                return None
            
            parts = line.split(",")
            rssi = float(parts[3])
            
            n_pairs = len(csi_values) // 2
            imag = np.array(csi_values[0::2][:n_pairs], dtype=np.float32)
            real = np.array(csi_values[1::2][:n_pairs], dtype=np.float32)
            amplitude = np.sqrt(real**2 + imag**2)
            phase = np.arctan2(imag, real)
            
            self.frame_count += 1
            self.amplitude_history.append(amplitude)
            self.phase_history.append(phase)
            self.rssi_history.append(rssi)
            recent_amps = np.array(list(self.amplitude_history)[-self.smoothing_window:])
            
            if len(recent_amps) < 10:
                return self._get_status(calibrating=True)
            
            # Calibration phase
            if not self.is_calibrated:
                spatial_std = float(np.std(amplitude))
                self.calibration_buffer.append(spatial_std)
                
                if len(self.calibration_buffer) >= self.config["calibration_samples"]:
                    self.baseline_amplitude = float(np.median(list(self.calibration_buffer)))
                    self.is_calibrated = True
                    self.calibration_in_progress = False
                    logger.info(f"Calibrated: baseline={self.baseline_amplitude:.2f}")
                return self._get_status(calibrating=True)
            
            # Calculate metrics
            spatial_variance = float(np.std(amplitude))
            
            if len(recent_amps) >= 20:
                temporal_variance = float(np.var(np.mean(recent_amps, axis=1)))
            else:
                temporal_variance = 0.0
            
            if len(self.phase_history) >= 5:
                phase_changes = []
                for i in range(len(self.phase_history) - 1):
                    phase_changes.extend(np.diff(self.phase_history[i]))
                phase_rate = float(np.std(phase_changes)) if phase_changes else 0.0
            else:
                phase_rate = 0.0
            
            baseline = self.baseline_amplitude or 1.0
            raw_presence = min(1.0, spatial_variance / (baseline * 1.5))
            raw_movement = min(1.0, temporal_variance / (baseline * 0.5 + 0.1))
            combined_movement = raw_movement * 0.6 + phase_rate * 2.0 * 0.4
            
            self.smoothed_presence = self.ema_alpha * raw_presence + (1 - self.ema_alpha) * self.smoothed_presence
            self.smoothed_movement = self.ema_alpha * combined_movement + (1 - self.ema_alpha) * self.smoothed_movement
            self.smoothed_rssi = 0.1 * rssi + 0.9 * self.smoothed_rssi
            
            self.presence_value_history.append(self.smoothed_presence)
            presence_variance = float(np.var(list(self.presence_value_history)[-30:])) if len(self.presence_value_history) >= 30 else 0.0
            movement_detected = self.smoothed_movement > self.movement_threshold
            
            # Get average amplitude and add to fall buffer
            avg_amplitude = float(np.mean(amplitude))
            self.fall_buffer.append(avg_amplitude)
            
            # Perform fall detection
            fall_risk, fall_detected = self._detect_fall(self.fall_buffer)
            
            # Classify movement type
            movement_type, risk_level = self._classify_movement_type(fall_risk, movement_detected)
            
            # Store for API
            self.last_movement_type = movement_type
            self.last_risk_level = risk_level
            
            # Stability and position
            stability = max(0, 1.0 - (presence_variance * 2000))
            position = self._estimate_position(amplitude, phase)
            self.position_history.append(position)
            spatial_profile = self._get_spatial_profile(amplitude)
            
            return self._get_status(
                calibrating=False,
                spatial_profile=spatial_profile,
                timestamp=time.time(),
                presence_variance=presence_variance,
                stability=stability,
                movement_type=movement_type,
                risk_level=risk_level,
                fall_detected=fall_detected
            )
            
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None
    
    def _estimate_position(self, amplitude, phase):
        n = len(amplitude)
        phase_gradient = np.gradient(phase[-n//2:]) if len(phase) >= n//2 else 0
        amp_mean = np.mean(amplitude) + 1e-6
        amp_norm = amplitude / amp_mean
        weights = np.linspace(0, 1, n)
        weighted_pos = np.sum(weights * amp_norm) / (n + 1e-6)
        phase_weight = float(np.mean(np.abs(phase_gradient)) * 10)
        return float(np.clip(weighted_pos * 0.7 + phase_weight * 0.3, 0, 1))
    
    def _get_spatial_profile(self, amplitude):
        n = len(amplitude)
        bin_size = max(1, n // self.spatial_bins)
        profile = []
        for i in range(self.spatial_bins):
            start = i * bin_size
            end = start + bin_size if i < self.spatial_bins - 1 else n
            profile.append(float(np.mean(amplitude[start:end])))
        if profile:
            mean_val = np.mean(profile)
            if mean_val > 0:
                profile = [v / mean_val for v in profile]
        return profile
    
    def _get_status(self, calibrating, spatial_profile=None, timestamp=None,
                   presence_variance=None, stability=None, movement_type=None,
                   risk_level=0.0, fall_detected=False):
        
        # Use stored values as fallback
        if movement_type is None:
            movement_type = getattr(self, 'last_movement_type', "NORMAL")
        if risk_level == 0.0:
            risk_level = getattr(self, 'last_risk_level', 0.0)
        
        positions = list(self.position_history)
        avg_position = float(np.mean(positions)) if positions else 0.5
        direction = 0
        if len(positions) >= 5:
            recent = np.mean(positions[-3:])
            older = np.mean(positions[-6:-3])
            direction = 1 if recent > older + 0.05 else (-1 if recent < older - 0.05 else 0)
        
        return {
            "type": "update",
            "timestamp": timestamp or time.time(),
            "calibrating": calibrating or self.calibration_in_progress,
            "calibration_progress": min(1.0, len(self.calibration_buffer) / self.config["calibration_samples"]),
            "baseline": round(self.baseline_amplitude, 2) if self.baseline_amplitude else None,
            "presence": bool(presence_variance > self.presence_variance_threshold) if presence_variance is not None else False,
            "presence_value": round(self.smoothed_presence, 4),
            "presence_variance": round(presence_variance, 6) if presence_variance else 0,
            "stability": round(stability, 4) if stability else 1.0,
            "movement": self.smoothed_movement > self.movement_threshold,
            "movement_value": round(self.smoothed_movement, 4),
            "movement_type": movement_type,
            "risk_level": round(risk_level, 3),
            "fall_detected": fall_detected,
            "position": round(avg_position, 3),
            "position_raw": positions[-1] if positions else 0.5,
            "direction": direction,
            "spatial_profile": spatial_profile or [1.0] * self.spatial_bins,
            "rssi": round(self.smoothed_rssi, 1),
            "stable": len(self.amplitude_history) >= self.smoothing_window,
            "frame_count": self.frame_count,
        }


app = FastAPI(title="CSI Light Dashboard")

class WSManager:
    def __init__(self):
        self.connections = []

ws_manager = WSManager()

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/status")
async def get_status():
    if processor:
        return processor._get_status(not processor.is_calibrated)
    return {"error": "Not initialized"}

@app.post("/calibrate")
async def calibrate():
    if processor:
        processor.start_calibration()
        return {"status": "ok", "message": "Calibration started. Please ensure room is empty."}
    return {"status": "error"}

@app.get("/calibration-status")
async def calibration_status():
    if processor:
        return {
            "is_calibrated": processor.is_calibrated,
            "calibration_in_progress": processor.calibration_in_progress,
            "baseline": round(processor.baseline_amplitude, 2) if processor.baseline_amplitude else None,
            "samples_collected": len(processor.calibration_buffer),
            "samples_needed": CONFIG["calibration_samples"],
        }
    return {"error": "Not initialized"}

@app.post("/threshold")
async def set_threshold(presence: float = 0.0005, movement: float = 1.0):
    if processor:
        processor.presence_variance_threshold = presence
        processor.movement_threshold = movement
        return {"status": "ok", "presence_variance_threshold": presence, "movement_threshold": movement}
    return {"status": "error"}

@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    ws_manager.connections.append(ws)
    try:
        while running:
            await ws.receive_text()
    except:
        if ws in ws_manager.connections:
            ws_manager.connections.remove(ws)

async def broadcast(data):
    for ws in ws_manager.connections[:]:
        try:
            await ws.send_json(data)
        except:
            if ws in ws_manager.connections:
                ws_manager.connections.remove(ws)

running = True
processor = None
serial_queue = queue.Queue()
serial_reader_thread = None

async def collection_loop():
    global running, processor
    parse_count = 0
    while running:
        try:
            line = serial_queue.get(timeout=0.1)
            if line:
                status = processor.process(line)
                if status:
                    parse_count += 1
                    status["parse_count"] = parse_count
                    await broadcast(status)
        except queue.Empty:
            pass
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup():
    global processor, serial_reader_thread
    import threading
    
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(CONFIG["db_path"])
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS detection_log
        (id INTEGER PRIMARY KEY, timestamp REAL, presence INTEGER, movement INTEGER,
         position REAL, direction INTEGER, rssi REAL, presence_value REAL,
         movement_value REAL, spatial_profile TEXT)""")
    conn.commit()
    conn.close()
    
    processor = CSIProcessor(CONFIG)
    
    reader = CSIReader(CONFIG["serial_port"], CONFIG["baud_rate"], serial_queue)
    serial_reader_thread = threading.Thread(target=reader.run, daemon=True)
    serial_reader_thread.start()
    
    asyncio.create_task(collection_loop())
    logger.info("CSI Light Dashboard started on port 8082")

@app.on_event("shutdown")
async def shutdown():
    global running
    running = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
