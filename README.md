# Vizzy: Enhancing Robotic Manipulation with Vision-Enabled Large Language Models

**Authors:** Burke Dambly, Miles Bovero, Lawrence Ponce, Brian Stites, Jesse Jenkins  
**Affiliation:** University of Utah — Electrical and Computer Engineering Capstone Project  

---

## Overview

Vizzy is a modular 5-axis robotic arm designed to demonstrate the integration of computer vision, inverse kinematics, and large language model (LLM) reasoning for adaptive robotic control.  
The project explores how artificial intelligence can make robotic systems more flexible and user-friendly by combining visual perception and semantic reasoning into a single integrated robotic platform.

Unlike traditional robotic systems that require explicit programming and rigid motion paths, Vizzy enables high-level natural language interaction. Through vision-based object detection and LLM-driven task scheduling, the system can autonomously interpret and execute user requests.

<p align="center">
  <img src="docs/diagrams/VizzySystemDiagram.jpeg" 
       alt="Vizzy System Diagram" 
       width="800" 
       style="border:3px solid white; border-radius:6px;">
</p>

<p align="center"><em>Figure 1 — Conceptual overview of Vizzy’s modular arm system.</em></p>

---

## System Architecture

### Hardware

- **Motors:** MJBOTS mj5208 BLDC outrunner motors  
- **Motor Controllers:** Moteus r.11 (Field-Oriented Control)  
- **Microcontrollers:** Raspberry Pi 4B and Raspberry
- **Sensors:** 16MP USB camera (YOLO11 vision model) and VL53L1X time-of-flight sensor  
- **Communication:** CAN-FD network using the Moteus protocol  
- **Structure:** CNC-milled T2-grade aluminum and PETG-CF 3D-printed components  
- **Power Supply:** Mean Well 24V .625A DC Power Supply 

<!-- 3x2 image grid for README -->
<table align="center">
  <tr>
    <td align="center">
      <img src="docs/images/C1Gearbox.jpg" width="250" height="250"
           style="object-fit:cover;border:3px solid white;border-radius:6px;" alt="Arm Assembly">
      <div style="font-size:12px;color:#999;margin-top:6px;">2-Stage Planetary GearBox</div>
    </td>
    <td align="center">
      <img src="docs/images/gripper.jpg" width="250" height="250"
           style="object-fit:cover;border:3px solid white;border-radius:6px;" alt="Joint Close-Up">
      <div style="font-size:12px;color:#999;margin-top:6px;">End Effector</div>
    </td>
    <td align="center">
      <img src="docs/CAD/screenshots/UpperArm.jpg" width="250" height="250"
           style="object-fit:cover;border:3px solid white;border-radius:6px;" alt="Moteus Controllers">
      <div style="font-size:12px;color:#999;margin-top:6px;">Upper Arm Linkage</div>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/images/ArmOnDesk.jpg" width="250" height="250"
           style="object-fit:cover;border:3px solid white;border-radius:6px;" alt="Electronics Hub">
      <div style="font-size:12px;color:#999;margin-top:6px;">System with Power Box</div>
    </td>
    <td align="center">
      <img src="docs/images/GripperHolding.jpg" width="250" height="250"
           style="object-fit:cover;border:3px solid white;border-radius:6px;" alt="Camera Mount">
      <div style="font-size:12px;color:#999;margin-top:6px;">Printed End Effector</div>
    </td>
    <td align="center">
      <img src="docs/images/ArmExtended.jpeg" width="250" height="250"
           style="object-fit:cover;border:3px solid white;border-radius:6px;" alt="Power System">
      <div style="font-size:12px;color:#999;margin-top:6px;">Arm Extended with Aluminum</div>
    </td>
  </tr>
</table>

<p align="center"><em>Figure 2 — Hardware overview showing mechanical components.</em></p>


### Software

- **Vision Stack:** YOLO11 segmentation for object recognition and localization  
- **Semantic Layer:** LLM-based contextual enrichment of detected objects  
- **Control Framework:** Python API for inverse kinematics and motion planning  
- **GUI:** Tkinter-based interface for manual coordinate input and visualization  
- **Embedded Firmware:** FreeRTOS task scheduling for low-level actuation and safety  

![Placeholder: Software Architecture Diagram](asf.jpeg)  
*Figure 3 — Software architecture illustrating perception, reasoning, and control flow.*

---

## Key Features

- Five-axis robotic manipulation with inverse kinematics control  
- Real-time visual perception using YOLO11 segmentation  
- Natural-language task execution via OpenAI’s API  
- Modular structure with customizable end-effectors  
- Safety-limited motion and emergency fault handling  
- Open-source software stack for reproducibility and community use  

![Placeholder: Vizzy Key Features Collage](images/placeholder_features.png)  
*Figure 4 — Visualization of major system capabilities.*

---

## Evaluation Summary

Vizzy was designed to validate the feasibility of LLM-assisted robotic control in constrained environments.  
Testing focused on three major domains:

1. **Mechanical Performance** — Verified torque capacity of mj5208 BLDC motors with custom 9:1 PETG-CF planetary gearboxes.  
2. **Perception and Sensing** — Achieved real-time segmentation (~5 ms inference) using YOLO11 with 16MP USB camera and VL53L1X ToF sensor.  
3. **Software Integration** — Implemented JSONL-based communication between vision (laptop) and control (Raspberry Pi) and validated natural-language task scheduling via OpenAI API.

![Placeholder: Vizzy Testing Setup](images/placeholder_testing.png)  
*Figure 7 — Prototype testing environment with motor torque and vision pipeline validation.*

---

## Future Work

- Integrate full depth-fusion perception for 3D object localization  
- Implement instance-level memory and tracking across frames  
- Develop more advanced grasping end-effectors  
- Optimize latency and reliability of the LLM interface  
- Add obstacle-aware path planning for dynamic environments  

![Placeholder: Vizzy Future Concepts](images/placeholder_future.png)  
*Figure 8 — Concept render for next iteration including multi-sensor fusion.*

---

## Acknowledgements

This project was developed as part of the University of Utah Electrical and Computer Engineering Capstone Program.  
Special thanks to the faculty mentors and the Utah Machine Shop for technical support and machining resources.


