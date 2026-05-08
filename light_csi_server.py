#!/usr/bin/env python3
"""Lightweight CSI Dashboard - Presence & Movement Detection"""
import time, asyncio, logging, queue
from pathlib import Path
from collections import deque
import numpy as np
import serial
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import uvicorn
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csi_light")

CONFIG = {
    "serial_port": "/dev/ttyACM0",
    "baud_rate": 115200,
    "smoothing_window": 50,
    "ema_alpha": 0.1,
    "presence_threshold": 0.3,
    "movement_threshold": 1.0,
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
        """Run in separate thread - reads raw bytes and extracts CSI lines"""
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
                except Exception as e:
                    pass
                time.sleep(0.01)
                    
            ser.close()
        except Exception as e:
            logger.error(f"Serial reader error: {e}")
    
    def _extract_csi_lines(self):
        """Extract complete CSI_DATA lines from buffer"""
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
        self.presence_threshold = config["presence_threshold"]
        self.movement_threshold = config["movement_threshold"]
        self.spatial_bins = config["spatial_bins"]
        self.amplitude_history = deque(maxlen=self.smoothing_window * 2)
        self.phase_history = deque(maxlen=self.smoothing_window)
        self.rssi_history = deque(maxlen=100)
        self.smoothed_presence = 0.0
        self.smoothed_movement = 0.0
        self.smoothed_rssi = -50.0
        self.calibration_buffer = deque(maxlen=config["calibration_samples"])
        self.is_calibrated = False
        self.baseline_amplitude = None
        self.position_history = deque(maxlen=50)
        self.frame_count = 0
        self.calibration_in_progress = False
        
    def start_calibration(self):
        """Manually start calibration (reset and recalibrate)"""
        self.calibration_buffer.clear()
        self.is_calibrated = False
        self.baseline_amplitude = None
        self.calibration_in_progress = True
        self.smoothed_presence = 0.0
        self.smoothed_movement = 0.0
        logger.info("Calibration started")
        
    def process(self, line):
        """Process a CSI_DATA line - robust parsing"""
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
            mac = parts[2].strip()
            
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
            
            position = self._estimate_position(amplitude, phase)
            self.position_history.append(position)
            spatial_profile = self._get_spatial_profile(amplitude)
            
            return self._get_status(calibrating=False, spatial_profile=spatial_profile, timestamp=time.time())
            
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
        
    def _get_status(self, calibrating, spatial_profile=None, timestamp=None):
        presence = self.smoothed_presence > self.presence_threshold
        movement = self.smoothed_movement > self.movement_threshold
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
            "presence": bool(presence),
            "movement": bool(movement),
            "presence_value": round(self.smoothed_presence, 4),
            "movement_value": round(self.smoothed_movement, 4),
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
    """Manually trigger calibration"""
    if processor:
        processor.start_calibration()
        return {"status": "ok", "message": "Calibration started. Please ensure room is empty."}
    return {"status": "error", "message": "Processor not initialized"}

@app.get("/calibration-status")
async def calibration_status():
    """Get calibration details"""
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
async def set_threshold(presence: float = 0.3, movement: float = 1.0):
    """Update detection thresholds"""
    if processor:
        processor.presence_threshold = presence
        processor.movement_threshold = movement
        return {"status": "ok", "presence_threshold": presence, "movement_threshold": movement}
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
    """Process serial data from queue"""
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