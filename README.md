# Sync Tank

![Sync Tank Banner](images/sync.jpg)

**Sync Tank** is an open-source platform for intelligent aquarium monitoring, enrichment, and interaction.  
It is built to combine hardware at the tank (Raspberry Pi, cameras, servos, sensors) with a server-side processing pipeline (computer vision, AI models) — giving aquarists, hobbyists, and educators access to the hidden world inside aquatic environments.

We built Sync Tank to go beyond just looking at the tank. It’s about **interacting, enriching, and understanding** what’s happening beneath the surface.

---

## What Sync Tank Does

- **Real-time FPV inspection:**  
  View your aquarium live through FPV goggles or a browser, using Pi Cameras or USB endoscopic cameras.

- **Real-time monitoring and alerts:**  
  Environmental sensors track temperature, light, motion; CV models identify fish, shrimp, turtles, lizards in the tank.

- **Robotic enrichment:**  
  Control MG995 servo-driven arms (REEFLEX) to rearrange the habitat or deliver feeding, either manually or on schedule.

- **Education and research:**  
  Understand species interactions, population behaviors, and environmental dynamics; use Sync Tank as an educational tool in classrooms, labs, or hobbyist workshops.

---

## Core Components

- **SEE SEA TV:**  
  Multi-camera streaming system, Raspberry Pi 5 with PiCamera (extended CSI cables) + USB endoscopic cams.  
  We pipe the stream live to local servers, FPV goggles, or browser displays.

- **REEFLEX:**  
  Physical enrichment system, MG995 servo motors + PCA9685 I2C driver board + 3D-printed tools.  
  We can move habitat objects, deliver food, or interact with tank residents.

- **DEEPLINK:**  
  Computer vision + AI recognition pipeline.  
  We run YOLOv8 object detection on stills sent from the tank, optionally passing results to an Ollama `llava:7b` multimodal model for natural language feedback.

---

## Hardware

- Raspberry Pi 5  
- Pi Camera v3 (extended cables) + USB endoscope cams  
- MG995 servo motors with PCA9685 driver  
- Sensors: temperature (DS18B20), light, water level, motion  
- Power: 5V for Pi, 5–7.2V for servos  
- Server PC: AMD CPU, NVIDIA GPU (tested on RTX 3060)  
- 3D-printed mounts and feeder systems (open-source STLs provided)

---

## What We Used (Open and Transparent)

- **Raspberry Pi 5** on the tank side, running minimal Python scripts for camera feed and servo control.
- **Ultralytics YOLOv8** for object detection, recognizing common tank residents (fish, shrimp, turtles, lizards).
- **Ollama llava:7b** multimodal model, giving human-readable explanations of what’s seen.
- **3D-printed REEFLEX parts**: freely shared designs for mounting arms, feeding tools, and camera holders.

---

## Development Pain Points

- **Video latency:**  
  Pushing live streams through Raspberry Pi over Wi-Fi, especially from multiple cameras, creates unavoidable lag. We improved this by prioritizing wired connections or local servers.

- **Servo noise and heat:**  
  MG995 servos under continuous load heat up fast and generate noise; we tuned pulse widths and implemented cooldown periods.

- **AI model bottlenecks:**  
  YOLOv8 and llava:7b run best on a discrete GPU. Attempting local inference on the Raspberry Pi itself was a hard no-go. All heavy compute is offloaded to the server.

- **Networking fragility:**  
  Sending video, sensor data, and control signals over home networks (especially through 4G LTE hotspots) introduced connection drops and occasional out-of-sync states.

---

## What Sync Tank Makes Possible

- Inspect tanks live in first person, from inside-out, using FPV goggles.
- Count and monitor how many fish or shrimp are active at any time.
- Enrich habitat with moving structures or automatic feeding.
- Provide classroom demos to teach about ecosystems and animal behavior.
- Expand hobbyist setups into automated, intelligent micro-ecosystems.
- Collect time-lapse data on growth, health, or environmental change.

---

## Open Resources

- YOLOv8 models: https://github.com/ultralytics/ultralytics  
- Ollama llava:7b: https://ollama.ai/  
- Raspberry Pi: https://www.raspberrypi.com/  
- PCA9685 Servo Driver: https://www.adafruit.com/product/815  
- Open 3D Prints (coming soon): [3d_prints/ directory]

---

## License

MIT License.  
We believe in open science, open hardware, and open curiosity.  
Please feel free to fork, improve, and remix — and share what you discover.

---

## Acknowledgments

Thanks to the robotics, maker, aquarist, and open-source communities who inspired this project.  
Sync Tank stands on the shoulders of your ideas, tools, and experiments.
