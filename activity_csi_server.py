#!/usr/bin/env python3
"""Enhanced CSI Dashboard - Full Activity Detection Pipeline

Activities: Still, Breathing, Wake up, Walking, Moving furniture, Falling

Pipeline:
  ESP32 TX -> ESP32 RX -> CSI Buffer -> Preprocessing -> Features -> Classifier -> Label

Preprocessing:
  - Phase sanitization (conjugate multiplication)
  - Bandpass filter: 0.1-50 Hz
  - Subcarrier selection: 10-15 highest-variance
  - Sliding window: 2s, 50% overlap

Features:
  - Amplitude mean / std / variance
  - FFT dominant frequency + band energy
  - Wavelet energy (DWT levels 3-5)
  - Autocorrelation peak
  - Zero-crossing rate
"""

import time, asyncio, logging, queue, json, threading
from pathlib import Path
from collections import deque
from datetime import datetime
import numpy as np
import serial
from scipy import signal
from scipy.signal import butter, filtfilt
try:
    import pywt
    HAS_PYWAVELETS = True
except ImportError:
    HAS_PYWAVELETS = False
    logging.warning("PyWavelets not available - wavelet features disabled")

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import uvicorn
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csi_activity")

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    "serial_port": "/dev/ttyACM0",
    "baud_rate": 115200,
    
    # Preprocessing
    "bandpass_low": 0.1,
    "bandpass_high": 50.0,
    "num_selected_subcarriers": 12,
    
    # Windowing
    "window_seconds": 2.0,
    "window_overlap": 0.5,
    "sample_rate": 100,  # CSI samples per second (estimated)
    
    # Activity-specific windows (seconds)
    "still_window": 3.0,
    "breathing_window": 5.0,
    "wakeup_window": 4.0,
    "walking_window": 2.0,
    "furniture_window": 2.0,
    "falling_window": 0.5,
    "falling_confirm_window": 2.0,
    
    # Thresholds (tuned for your setup)
    "still_variance_threshold": 0.001,
    "breathing_freq_low": 0.2,
    "breathing_freq_high": 0.5,
    "breathing_energy_threshold": 0.01,
    "wakeup_rise_threshold": 0.05,
    "wakeup_rise_time": 4.0,
    "walking_variance_threshold": 0.1,
    "walking_freq_low": 1.0,
    "walking_freq_high": 3.0,
    "walking_periodicity_threshold": 0.5,
    "furniture_variance_threshold": 0.5,
    "furniture_energy_threshold": 1.0,
    "falling_spike_threshold": 2.0,
    "falling_still_threshold": 0.01,
    "falling_confirm_time": 2.0,
    
    # Calibration
    "calibration_samples": 200,
    "spatial_bins": 8,
    
    # Training data collection
    "collect_training_data": True,
    "training_data_buffer_max": 10000,
    
    "db_path": "data/activity_csi.db",
    "training_data_path": "data/training_data",
}

# Activity labels
ACTIVITIES = ["still", "breathing", "wakeup", "walking", "furniture", "falling", "unknown"]


# ============================================
# CSI READER
# ============================================
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
                except Exception as e:
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


# ============================================
# CSI PROCESSOR WITH ACTIVITY DETECTION
# ============================================
class CSIProcessor:
    def __init__(self, config):
        self.config = config
        self.spatial_bins = config["spatial_bins"]
        self.n_subcarriers = config["num_selected_subcarriers"]
        
        # Sample rate estimation
        self.sample_times = deque(maxlen=1000)
        self.last_sample_time = None
        self.estimated_sample_rate = 50  # Hz
        
        # Bandpass filter (depends on sample_rate, so must init after)
        self.filter_coeffs = None
        self._init_filter()
        
        # Signal buffers for windowing
        self.amplitude_buffer = deque(maxlen=int(config["window_seconds"] * config["sample_rate"] * 2))
        self.phase_buffer = deque(maxlen=int(config["window_seconds"] * config["sample_rate"] * 2))
        self.rssi_buffer = deque(maxlen=1000)
        
        # Selected subcarrier indices (based on variance)
        self.selected_indices = None
        self.subcarrier_variance = None
        
        # Calibration
        self.calibration_buffer = deque(maxlen=config["calibration_samples"])
        self.is_calibrated = False
        self.baseline_amplitude = None
        self.frame_count = 0
        
        # Activity state tracking
        self.current_activity = "unknown"
        self.activity_confidence = 0.0
        self.activity_duration = 0.0
        self.last_activity_change = time.time()
        self.activity_history = deque(maxlen=50)
        
        # Specific activity detection state
        self.breathing_history = deque(maxlen=100)
        self.breathing_waveform = deque(maxlen=int(config["breathing_window"] * config["sample_rate"]))
        self.wakeup_baseline = None
        self.wakeup_start_time = None
        self.walking_periodicity_buffer = deque(maxlen=50)
        self.falling_spike_detected = False
        self.falling_spike_time = None
        self.furniture_energy_history = deque(maxlen=20)
        
        # Training data collection
        self.training_data_buffer = []
        self.collect_training_data = config["collect_training_data"]
        self.training_label = "unknown"
        self.training_start_time = None
        
        # Subcarrier statistics for selection
        self.subcarrier_sum = None
        self.subcarrier_count = 0
        
    def _init_filter(self):
        """Initialize bandpass filter"""
        try:
            nyquist = self.estimated_sample_rate / 2
            low = self.config["bandpass_low"] / nyquist
            high = min(self.config["bandpass_high"] / nyquist, 0.99)
            if low >= high:
                low = 0.001
            b, a = butter(4, [low, high], btype='band')
            self.filter_coeffs = (b, a)
            logger.info(f"Bandpass filter initialized: {self.config['bandpass_low']}-{self.config['bandpass_high']} Hz")
        except Exception as e:
            logger.warning(f"Filter initialization failed: {e}")
            self.filter_coeffs = None
    
    def set_training_label(self, label):
        """Set the label for training data collection"""
        if label in ACTIVITIES:
            self.training_label = label
            self.training_start_time = time.time()
            logger.info(f"Training data collection started for label: {label}")
        else:
            logger.warning(f"Invalid training label: {label}")
    
    def stop_training_collection(self):
        """Stop training data collection and save"""
        self.collect_training_data = False
        if self.training_data_buffer:
            self._save_training_data()
    
    def _save_training_data(self):
        """Save collected training data to file"""
        if not self.training_data_buffer:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config['training_data_path']}/{self.training_label}_{timestamp}.json"
        
        Path(self.config["training_data_path"]).mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump({
                "label": self.training_label,
                "timestamp": timestamp,
                "sample_count": len(self.training_data_buffer),
                "features": self.training_data_buffer
            }, f)
        
        logger.info(f"Saved {len(self.training_data_buffer)} samples to {filename}")
        self.training_data_buffer = []
    
    def _apply_bandpass_filter(self, data):
        """Apply bandpass filter to signal"""
        if self.filter_coeffs is None or len(data) < 10:
            return data
        
        try:
            filtered = filtfilt(self.filter_coeffs[0], self.filter_coeffs[1], data)
            return filtered
        except Exception as e:
            logger.debug(f"Filter error: {e}")
            return data
    
    def _sanitize_phase(self, amplitude, phase):
        """Phase sanitization using conjugate multiplication between adjacent subcarriers"""
        if len(amplitude) < 2:
            return amplitude, phase
        
        sanitized = np.zeros_like(amplitude)
        sanitized[0] = amplitude[0]
        
        for i in range(1, len(amplitude)):
            sanitized[i] = np.sqrt(np.abs(amplitude[i] * amplitude[i-1]))
        
        return sanitized, phase
    
    def _select_top_subcarriers(self, amplitude):
        """Select 10-15 highest variance subcarriers"""
        if self.selected_indices is not None and len(self.selected_indices) >= self.n_subcarriers:
            return amplitude[self.selected_indices]
        
        if self.subcarrier_variance is not None and len(self.subcarrier_variance) == len(amplitude):
            variances = self.subcarrier_variance
        else:
            variances = amplitude
        
        if len(variances) <= self.n_subcarriers:
            return amplitude
        
        sorted_indices = np.argsort(variances)[-self.n_subcarriers:]
        self.selected_indices = sorted_indices
        
        return amplitude[sorted_indices]
    
    def _update_subcarrier_stats(self, amplitude):
        """Update running statistics for subcarrier selection"""
        squared = amplitude ** 2
        if self.subcarrier_sum is None:
            self.subcarrier_sum = squared
        else:
            self.subcarrier_sum += squared
        self.subcarrier_count += 1
        
        if self.subcarrier_count % 500 == 0 and self.subcarrier_count > 0:
            mean_squares = self.subcarrier_sum / self.subcarrier_count
            self.subcarrier_variance = mean_squares - (np.mean(self.amplitude_buffer, axis=0) ** 2 if len(self.amplitude_buffer) > 0 else 0)
    
    def _extract_features(self, amplitude_window, phase_window=None):
        """Extract all features from a window"""
        features = {}
        
        # Amplitude features
        features['amp_mean'] = float(np.mean(amplitude_window))
        features['amp_std'] = float(np.std(amplitude_window))
        features['amp_var'] = float(np.var(amplitude_window))
        features['amp_max'] = float(np.max(amplitude_window))
        features['amp_min'] = float(np.min(amplitude_window))
        features['amp_range'] = features['amp_max'] - features['amp_min']
        
        # Zero-crossing rate
        mean_centered = amplitude_window - np.mean(amplitude_window)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(mean_centered)))) / 2
        features['zcr'] = float(zero_crossings / len(amplitude_window))
        
        # FFT features
        n = len(amplitude_window)
        fft_vals = np.fft.fft(amplitude_window)
        fft_freqs = np.fft.fftfreq(n, 1.0/self.estimated_sample_rate)
        
        positive_mask = fft_freqs > 0
        magnitude = np.abs(fft_vals[positive_mask])
        freqs = fft_freqs[positive_mask]
        
        if len(magnitude) > 0:
            # Dominant frequency
            dom_idx = np.argmax(magnitude)
            features['fft_dom_freq'] = float(freqs[dom_idx]) if dom_idx < len(freqs) else 0.0
            features['fft_dom_mag'] = float(magnitude[dom_idx])
            
            # Band energies
            band_0_1 = (freqs >= 0) & (freqs <= 1)
            band_1_5 = (freqs >= 1) & (freqs <= 5)
            band_5_20 = (freqs >= 5) & (freqs <= 20)
            
            features['fft_energy_0_1'] = float(np.sum(magnitude[band_0_1]**2))
            features['fft_energy_1_5'] = float(np.sum(magnitude[band_1_5]**2))
            features['fft_energy_5_20'] = float(np.sum(magnitude[band_5_20]**2))
        else:
            features['fft_dom_freq'] = 0.0
            features['fft_dom_mag'] = 0.0
            features['fft_energy_0_1'] = 0.0
            features['fft_energy_1_5'] = 0.0
            features['fft_energy_5_20'] = 0.0
        
        # Autocorrelation peak
        if len(amplitude_window) > 10:
            autocorr = np.correlate(amplitude_window, amplitude_window, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.1, distance=5)
            if len(peaks) > 0:
                features['autocorr_peak'] = float(peaks[0] + 1)
                features['autocorr_height'] = float(autocorr[peaks[0] + 1])
            else:
                features['autocorr_peak'] = 0.0
                features['autocorr_height'] = 0.0
        else:
            features['autocorr_peak'] = 0.0
            features['autocorr_height'] = 0.0
        
        # Wavelet features (DWT levels 3-5)
        if HAS_PYWAVELETS and len(amplitude_window) >= 16:
            try:
                wavelet_data = amplitude_window - np.mean(amplitude_window)
                coeffs = pywt.wavedec(wavelet_data, 'db4', level=4)
                
                features['wavelet_d3'] = float(np.sum(coeffs[3]**2)) if len(coeffs) > 3 else 0.0
                features['wavelet_d4'] = float(np.sum(coeffs[4]**2)) if len(coeffs) > 4 else 0.0
                features['wavelet_d5'] = float(np.sum(coeffs[5]**2)) if len(coeffs) > 5 else 0.0
            except Exception:
                features['wavelet_d3'] = 0.0
                features['wavelet_d4'] = 0.0
                features['wavelet_d5'] = 0.0
        else:
            features['wavelet_d3'] = 0.0
            features['wavelet_d4'] = 0.0
            features['wavelet_d5'] = 0.0
        
        return features
    
    def _detect_still(self, features, window_size_samples):
        """Detect still state: near-zero variance"""
        window_sec = window_size_samples / self.estimated_sample_rate
        if window_sec < self.config["still_window"]:
            return False, 0.0
        
        variance = features['amp_var']
        threshold = self.config["still_variance_threshold"]
        
        if variance < threshold:
            confidence = 1.0 - (variance / threshold) if threshold > 0 else 0.0
            confidence = max(0.0, min(1.0, confidence))
            return True, confidence
        return False, 0.0
    
    def _detect_breathing(self, features, window_size_samples):
        """Detect breathing: 0.2-0.5 Hz rhythm"""
        window_sec = window_size_samples / self.estimated_sample_rate
        if window_sec < self.config["breathing_window"]:
            return False, 0.0
        
        freq = features['fft_dom_freq']
        energy_0_1 = features['fft_energy_0_1']
        
        in_breathing_range = self.config["breathing_freq_low"] <= freq <= self.config["breathing_freq_high"]
        has_energy = energy_0_1 > self.config["breathing_energy_threshold"]
        
        if in_breathing_range and has_energy:
            confidence = min(1.0, energy_0_1 / 0.1)
            return True, confidence
        return False, 0.0
    
    def _detect_wakeup(self, features):
        """Detect wake up: slow amplitude rise from still"""
        variance = features['amp_var']
        
        if self.wakeup_baseline is None:
            self.wakeup_baseline = variance
            self.wakeup_start_time = time.time()
            return False, 0.0
        
        rise = variance - self.wakeup_baseline
        elapsed = time.time() - self.wakeup_start_time
        
        if elapsed > self.config["wakeup_rise_time"] and variance < self.config["still_variance_threshold"] * 5:
            self.wakeup_baseline = variance
            self.wakeup_start_time = time.time()
        
        if rise > self.config["wakeup_rise_threshold"] and elapsed > 2.0:
            confidence = min(1.0, rise / 0.2)
            return True, confidence
        
        if elapsed > 10.0:
            self.wakeup_baseline = variance
        
        return False, 0.0
    
    def _detect_walking(self, features, window_size_samples):
        """Detect walking: 1-3 Hz periodic, high variance"""
        window_sec = window_size_samples / self.estimated_sample_rate
        if window_sec < self.config["walking_window"]:
            return False, 0.0
        
        variance = features['amp_var']
        freq = features['fft_dom_freq']
        autocorr_peak = features['autocorr_peak']
        
        has_variance = variance > self.config["walking_variance_threshold"]
        in_walking_range = self.config["walking_freq_low"] <= freq <= self.config["walking_freq_high"]
        
        has_periodicity = features['autocorr_height'] > self.config["walking_periodicity_threshold"]
        
        if has_variance and in_walking_range:
            confidence = min(1.0, variance / 0.5) * 0.7
            if has_periodicity:
                confidence += 0.3
            return True, confidence
        
        return False, 0.0
    
    def _detect_furniture(self, features, window_size_samples):
        """Detect moving furniture: high energy, irregular"""
        window_sec = window_size_samples / self.estimated_sample_rate
        if window_sec < self.config["furniture_window"]:
            return False, 0.0
        
        variance = features['amp_var']
        energy_5_20 = features['fft_energy_5_20']
        
        has_high_variance = variance > self.config["furniture_variance_threshold"]
        has_energy = energy_5_20 > self.config["furniture_energy_threshold"]
        
        if has_high_variance and has_energy:
            confidence = min(1.0, variance / 2.0)
            self.furniture_energy_history.append(variance)
            return True, confidence
        
        return False, 0.0
    
    def _detect_falling(self, features, current_amplitude):
        """Detect falling: sharp spike -> sudden stillness"""
        variance = features['amp_var']
        amplitude = features['amp_mean']
        
        now = time.time()
        
        if variance > self.config["falling_spike_threshold"]:
            self.falling_spike_detected = True
            self.falling_spike_time = now
            return False, 0.0
        
        if self.falling_spike_detected and self.falling_spike_time:
            elapsed = now - self.falling_spike_time
            
            if elapsed < self.config["falling_confirm_window"] and elapsed > 0.5:
                if variance < self.config["falling_still_threshold"]:
                    confidence = min(1.0, elapsed / self.config["falling_confirm_time"])
                    self.falling_spike_detected = False
                    return True, confidence
                else:
                    self.falling_spike_detected = False
        
        if self.falling_spike_detected and self.falling_spike_time:
            if now - self.falling_spike_time > 5.0:
                self.falling_spike_detected = False
        
        return False, 0.0
    
    def _classify_activity(self, features, window_size_samples, current_amplitude):
        """Classify current activity using algorithm-based detection"""
        detections = {}
        confidences = {}
        
        detections["still"], confidences["still"] = self._detect_still(features, window_size_samples)
        detections["breathing"], confidences["breathing"] = self._detect_breathing(features, window_size_samples)
        detections["wakeup"], confidences["wakeup"] = self._detect_wakeup(features)
        detections["walking"], confidences["walking"] = self._detect_walking(features, window_size_samples)
        detections["furniture"], confidences["furniture"] = self._detect_furniture(features, window_size_samples)
        falling_detected, falling_conf = self._detect_falling(features, current_amplitude)
        detections["falling"] = falling_detected
        confidences["falling"] = falling_conf
        
        # Priority: falling > furniture > walking > wakeup > breathing > still
        priority_order = ["falling", "furniture", "walking", "wakeup", "breathing", "still"]
        
        best_activity = "unknown"
        best_confidence = 0.0
        
        for activity in priority_order:
            if detections[activity] and confidences[activity] > best_confidence:
                best_activity = activity
                best_confidence = confidences[activity]
        
        # Update activity state
        if best_activity != self.current_activity:
            self.last_activity_change = time.time()
            self.current_activity = best_activity
            self.activity_history.append(best_activity)
        
        self.activity_duration = time.time() - self.last_activity_change
        self.activity_confidence = best_confidence
        
        return best_activity, best_confidence
    
    def process(self, line):
        """Process a CSI_DATA line"""
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
            now = time.time()
            self.sample_times.append(now)
            
            if len(self.sample_times) >= 2:
                time_diffs = np.diff(self.sample_times)
                valid_diffs = time_diffs[time_diffs > 0]
                if len(valid_diffs) > 0:
                    self.estimated_sample_rate = 1.0 / np.median(valid_diffs)
            
            amplitude, phase = self._sanitize_phase(amplitude, phase)
            
            self._update_subcarrier_stats(amplitude)
            
            amplitude = self._select_top_subcarriers(amplitude)
            
            avg_amplitude = float(np.mean(amplitude))
            self.amplitude_buffer.append(avg_amplitude)
            self.phase_buffer.append(float(np.mean(phase)))
            self.rssi_buffer.append(rssi)
            
            self.rssi_buffer.append(rssi)
            
            if not self.is_calibrated:
                spatial_std = float(np.std(amplitude))
                self.calibration_buffer.append(spatial_std)
                
                if len(self.calibration_buffer) >= self.config["calibration_samples"]:
                    self.baseline_amplitude = float(np.median(list(self.calibration_buffer)))
                    self.is_calibrated = True
                    logger.info(f"Calibrated: baseline={self.baseline_amplitude:.2f}")
                return self._get_status(calibrating=True)
            
            window_size = int(self.config["window_seconds"] * self.estimated_sample_rate)
            
            if len(self.amplitude_buffer) < window_size // 2:
                return self._get_status(calibrating=False)
            
            amplitude_window = np.array(list(self.amplitude_buffer)[-window_size:])
            
            if len(amplitude_window) < 10:
                return self._get_status(calibrating=False)
            
            amplitude_filtered = self._apply_bandpass_filter(amplitude_window)
            
            features = self._extract_features(amplitude_filtered)
            
            current_amplitude = amplitude_window
            
            activity, confidence = self._classify_activity(features, window_size, current_amplitude)
            
            if self.collect_training_data:
                features["timestamp"] = now
                features["activity"] = activity
                features["rssi"] = rssi
                features["mac"] = mac
                self.training_data_buffer.append(features)
                
                if len(self.training_data_buffer) >= self.config["training_data_buffer_max"]:
                    self._save_training_data()
            
            return self._get_status(calibrating=False, activity=activity, 
                                   confidence=confidence, features=features)
            
        except Exception as e:
            logger.debug(f"Process error: {e}")
            return None
    
    def _get_status(self, calibrating=False, activity="unknown", confidence=0.0, features=None):
        """Get current status for dashboard"""
        positions = list(self.phase_buffer)
        if len(positions) >= 2:
            phase_diff = np.diff(positions)
            positions_normalized = (positions - np.min(phase_diff)) / (np.ptp(phase_diff) + 1e-10)
            avg_position = float(np.mean(positions_normalized))
        else:
            avg_position = 0.5
        
        spatial_profile = [1.0] * self.spatial_bins
        
        status = {
            "timestamp": time.time(),
            "calibrating": calibrating,
            "is_calibrated": self.is_calibrated,
            "frame_count": self.frame_count,
            
            # Activity detection
            "activity": activity,
            "activity_confidence": round(confidence, 3),
            "activity_duration": round(self.activity_duration, 2),
            "activity_history": list(self.activity_history),
            
            # Activity indicators (for dashboard display)
            "still": activity == "still",
            "breathing": activity == "breathing",
            "wakeup": activity == "wakeup",
            "walking": activity == "walking",
            "furniture": activity == "furniture",
            "falling": activity == "falling",
            "unknown": activity == "unknown",
            
            # Additional metrics
            "sample_rate": round(self.estimated_sample_rate, 1),
            "rssi": round(np.mean(list(self.rssi_buffer)) if self.rssi_buffer else -50.0, 1),
            "position": round(avg_position, 3),
            "spatial_profile": spatial_profile,
        }
        
        if features:
            status["features"] = {
                "amp_mean": round(features.get("amp_mean", 0), 4),
                "amp_std": round(features.get("amp_std", 0), 4),
                "fft_dom_freq": round(features.get("fft_dom_freq", 0), 2),
                "fft_energy_0_1": round(features.get("fft_energy_0_1", 0), 4),
                "zcr": round(features.get("zcr", 0), 4),
            }
        
        return status


# ============================================
# FASTAPI APPLICATION
# ============================================
app = FastAPI(title="CSI Activity Dashboard")

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
        processor.is_calibrated = False
        processor.calibration_buffer.clear()
        return {"status": "ok", "message": "Calibration started"}
    return {"status": "error"}

@app.post("/training/start")
async def start_training(label: str = "unknown"):
    if processor:
        processor.set_training_label(label)
        return {"status": "ok", "message": f"Training data collection started for: {label}"}
    return {"status": "error"}

@app.post("/training/stop")
async def stop_training():
    if processor:
        processor.stop_training_collection()
        return {"status": "ok", "message": "Training data collection stopped and saved"}
    return {"status": "error"}

@app.get("/training/status")
async def training_status():
    if processor:
        return {
            "collecting": processor.collect_training_data,
            "current_label": processor.training_label,
            "sample_count": len(processor.training_data_buffer),
        }
    return {"error": "Not initialized"}

@app.get("/activities")
async def get_activities():
    return {"activities": ACTIVITIES}

running = True
processor = None
serial_queue = queue.Queue()
serial_reader_thread = None

async def broadcast(data):
    for ws in ws_manager.connections[:]:
        try:
            await ws.send_json(data)
        except:
            if ws in ws_manager.connections:
                ws_manager.connections.remove(ws)

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

@app.on_event("startup")
async def startup():
    global processor, serial_reader_thread, running
    
    Path("data").mkdir(exist_ok=True)
    Path(CONFIG["training_data_path"]).mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(CONFIG["db_path"])
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS activity_log 
        (id INTEGER PRIMARY KEY, timestamp REAL, activity TEXT, confidence REAL,
         still INTEGER, breathing INTEGER, wakeup INTEGER, walking INTEGER,
         furniture INTEGER, falling INTEGER, rssi REAL)""")
    conn.commit()
    conn.close()
    
    processor = CSIProcessor(CONFIG)
    
    reader = CSIReader(CONFIG["serial_port"], CONFIG["baud_rate"], serial_queue)
    serial_reader_thread = threading.Thread(target=reader.run, daemon=True)
    serial_reader_thread.start()
    
    asyncio.create_task(collection_loop())
    logger.info("CSI Activity Dashboard started on port 8082")

@app.on_event("shutdown")
async def shutdown():
    global running
    running = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
