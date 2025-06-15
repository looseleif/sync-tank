# Sync Tank

![Sync Tank Banner](images/sync.jpeg)

**Sync Tank** is an open-source platform for intelligent aquarium monitoring, enrichment, and interaction.  
Designed with modularity and scalability in mind, it connects hardware at the tank to remote compute systems—bridging live aquatic observation with robotics and computer vision.

> Water and low voltage have never made this much sense.

---

## Features

- Live video monitoring using Pi Camera (with extended flex cables) or USB endoscopic cameras  
- MG995 servo-driven robotics for physical interaction and enrichment  
- Environmental sensing (e.g., temperature, light, motion) with real-time feedback  
- Modular control and automation routines  
- Scalable architecture: from lightweight Python scripts to advanced computer vision pipelines  
- 3D printable mounts and enclosures for sensors, cameras, and actuators  

---

## Getting Started

### Hardware Requirements

**Tank-side device (client):**
- Raspberry Pi 5
- Pi Camera with extended flex cable and USB endoscopic camera  
- MG995 servo motors and driver board  
- Sensor modules (temperature, ambient light, proximity, etc.)  
- Wiring harness and power management (5V for Pi, 5–7.2V for servos)  
- 3D printed components for mechanical mounts and fixtures

**Server-side device (host):**
- Standard workstation PC (tested with AMD CPU and NVIDIA GPU)
- Acts as the main compute node for heavier processing, inference, and data storage
- Can be your home server, local desktop, or remote cloud compute depending on setup

---

### Software Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sync-tank.git
   cd sync-tank
