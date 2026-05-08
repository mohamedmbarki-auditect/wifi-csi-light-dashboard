# WiFi CSI Light Dashboard

A lightweight real-time WiFi Channel State Information (CSI) dashboard for presence and movement detection using ESP32 devices.

## Features

- **Real-time Detection**: Presence/No-Presence and Movement/Still indicators
- **Position Estimation**: Shows where a person/object is along the TX-RX path
- **Direction Tracking**: Indicates movement direction (toward or away from TX)
- **Spatial Profile**: CSI amplitude visualization across subcarrier regions
- **Smooth Output**: EMA-based smoothing for stable readings (unlike typical fluctuating dashboards)
- **WebSocket Streaming**: Fast 5Hz updates to browser
- **SQLite Storage**: Local database for historical data

## Requirements

- ESP32 device flashed with WiFi CSI firmware (ESP32-CSI-Tool or similar)
- Raspberry Pi or Linux machine with USB serial connection
- Python 3.8+
- FastAPI, Uvicorn, PySerial, NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/wifi-csi-light-dashboard.git
cd wifi-csi-light-dashboard

# Install dependencies
pip install fastapi uvicorn pyserial numpy

# Run
python light_csi_server.py
```

## Configuration

Edit the `CONFIG` dictionary in `light_csi_server.py`:

```python
CONFIG = {
    "serial_port": "/dev/ttyACM0",  # ESP32 serial port
    "baud_rate": 115200,
    "smoothing_window": 50,          # Larger = smoother
    "ema_alpha": 0.1,               # Lower = smoother (0.1-0.3)
    "presence_threshold": 0.3,      # Binary presence threshold
    "movement_threshold": 0.15,     # Binary movement threshold
    "calibration_samples": 200,     # Frames for baseline calibration
    "spatial_bins": 8,              # Subcarrier regions for spatial profile
    "db_path": "data/light_csi.db",
}
```

## Usage

1. Connect ESP32 (RX mode) to the Raspberry Pi via USB
2. Connect ESP32 (TX mode) to your laptop or another power source
3. Start the TX device broadcasting packets
4. Run: `python light_csi_server.py`
5. Open browser at `http://YOUR_PI_IP:8082`

## Hardware Setup

```
┌─────────┐    WiFi CSI     ┌─────────┐
│  TX     │ ────────────→  │  RX     │
│ ESP32   │   ~3 meters    │ ESP32   │
└─────────┘                └────┬────┘
                                 │
                           USB to Pi
```

## Dashboard Features

| Panel | Description |
|-------|-------------|
| **Detection Status** | Binary Presence/Movement with continuous values |
| **Position Estimation** | Visual indicator of person/object along path |
| **Spatial Profile** | Bar chart of CSI amplitude by subcarrier region |
| **Charts** | Historical presence probability and movement index |
| **Activity Log** | Timestamped detection events |

## API Endpoints

- `GET /` - Dashboard web interface
- `GET /status` - Current detection status JSON
- `WS /ws` - WebSocket for real-time updates

## Algorithm

The detection uses:
1. **Spatial Variance**: Standard deviation of CSI amplitude across subcarriers
2. **Temporal Variance**: Change in mean amplitude over time  
3. **Phase Rate**: Rate of phase changes (indicates movement)
4. **EMA Smoothing**: Exponential moving average for stable output

## License

MIT License

## Acknowledgments

- ESP32-CSI-Tool project for WiFi CSI firmware
- Built with FastAPI, uPlot, and modern async Python