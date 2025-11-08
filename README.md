<div align="center" style="margin-top:16px;margin-bottom:12px;">
  <h1 style="margin:0 0 6px 0;font-size:38px;line-height:1.15;letter-spacing:.3px;">
    Vizzy: Enhancing Robotic Manipulation with Vision-Enabled Large Language Models
  </h1>

  <div style="font-size:14px;color:#6a737d;margin-bottom:10px;">
    University of Utah — Electrical & Computer Engineering Capstone
  </div>

  <div style="font-size:15px;margin-bottom:14px;">
    Authors: Burke Dambly · Miles Bovero · Lawrence Ponce · Brian Stites · Jesse Jenkins
  </div>
  <br>
  <!-- tech badges -->
  <div style="margin-bottom:14px;">
    <img alt="Moteus" src="https://img.shields.io/badge/Made%20with-Moteus-1e40af.svg" />
    <img alt="YOLO" src="https://img.shields.io/badge/YOLO-v11-ef4444.svg?logo=yolo&logoColor=white" />
    <img alt="Raspberry Pi" src="https://img.shields.io/badge/Raspberry%20Pi-4-c51a4a.svg?logo=raspberrypi&logoColor=white" />
    <img alt="OpenAI API" src="https://img.shields.io/badge/OpenAI-API-7b3fe4.svg?logo=openai&logoColor=white" />
    <img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white" />
  </div>

  <hr style="margin:0 auto;border:0;height:1px;width:90%;background:linear-gradient(90deg,rgba(0,0,0,0),rgba(140,140,140,.35),rgba(0,0,0,0));" />
</div>



## Overview

<!-- Overview Section -->
<div style="margin-top:24px; margin-bottom:24px; padding:18px 22px; border-left:4px solid #2563eb; background:rgba(37,99,235,0.05); border-radius:8px;">
  <p style="margin:0 0 12px 0;">
    <strong>Vizzy</strong> is a modular 5-axis robotic arm designed to demonstrate the integration of 
    <strong>computer vision</strong>, <strong>inverse kinematics</strong>, and 
    <strong>large language model (LLM)</strong> reasoning for adaptive robotic control.  
    The project explores how artificial intelligence can make robotic systems more flexible and user-friendly 
    by combining visual perception and semantic reasoning into a single integrated robotic platform.
  </p>

  <p style="margin:0;">
    Unlike traditional robotic systems that require explicit programming and rigid motion paths, Vizzy enables 
    high-level natural language interaction. Through vision-based object detection and LLM-driven task scheduling, 
    the system can autonomously interpret and execute user requests.
  </p>
</div>

<p align="center">
  <img src="docs/diagrams/VizzySystemDiagram.jpeg" 
       alt="Vizzy System Diagram" 
       width="820" 
       style="border:3px solid white; border-radius:6px; box-shadow:0 2px 10px rgba(0,0,0,.15);">
</p>

<p align="center"><em>Figure 1 — Conceptual overview of Vizzy’s modular arm system.</em></p>

<hr style="margin-top:28px; margin-bottom:0; border:0; height:1px; background:linear-gradient(90deg,rgba(0,0,0,0),rgba(140,140,140,.35),rgba(0,0,0,0));" />


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
<h2 id="about-the-authors" align="center">About the Authors</h2>

<table style="width:100%;border-collapse:separate;border-spacing:18px 14px;">
  <tr align="center" valign="top">
    <td style="width:33%;border:1px solid #d0d7de;border-radius:10px;padding:16px;background:#fafbfc;">
      <strong>Burke Dambly</strong><br>
      <em style="color:#57606a;">Computer Engineering</em><br>
      Focused on control systems, embedded firmware, and integration of Moteus-controlled BLDC actuators with high-level Python logic.
    </td>
    <td style="width:33%;border:1px solid #d0d7de;border-radius:10px;padding:16px;background:#fafbfc;">
      <strong>Miles Bovero</strong><br>
      <em style="color:#57606a;">Mechanical Engineering</em><br>
      Blah blah YOLO
    </td>
    <td style="width:33%;border:1px solid #d0d7de;border-radius:10px;padding:16px;background:#fafbfc;">
      <strong>Lawrence Ponce</strong><br>
      <em style="color:#57606a;">Computer Engineering</em><br>
      LM blah blah
    </td>
    
  </tr>

  <tr align="center" valign="top">
    <td style="width:33%;border:1px solid #d0d7de;border-radius:10px;padding:16px;background:#fafbfc;">
      <strong>Brian Stites</strong><br>
      <em style="color:#57606a;">Electrical Engineering</em><br>
      Designed and assembled the power distribution and safety systems for the robotic arm.
    </td>
    <td style="width:33%;border:1px solid #d0d7de;border-radius:10px;padding:16px;background:#fafbfc;">
      <strong>Jesse Jenkins</strong><br>
      <em style="color:#57606a;">Computer Science</em><br>
      Contributed to the inverse kinematics (IK) framework for precise arm positioning and control.
    </td>
    <td style="width:33%;border:1px dashed #d0d7de;border-radius:10px;padding:16px;background:#fafbfc;">
      <strong>Neal Patwari</strong><br>
      <em style="color:#57606a;">Faculty Advisor</em><br>
      Provided guidance on system integration and research methodology throughout the Vizzy capstone project.
    </td>
  </tr>
</table>











## Acknowledgements

This project was developed as part of the University of Utah Electrical and Computer Engineering Capstone Program.  
Special thanks to the faculty mentors and the Utah Machine Shop for technical support and machining resources.


