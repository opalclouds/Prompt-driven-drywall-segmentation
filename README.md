# Prompt-Driven Segmentation of Drywall Defects using CLIPSeg

This project implements a **prompt-based image segmentation system** to automatically identify drywall defects such as **cracks** and **joints** using natural language queries.

Unlike traditional segmentation models that require separate models per class, this system uses a **vision-language model** that can segment different defect types based on the input prompt.

Example prompts:

- segment crack
- segment drywall joint

---

# Overview

Drywall defect detection is an important quality control task in construction. Manual inspection is slow, subjective, and error-prone.

This project demonstrates how **vision-language models can automate inspection using flexible text prompts**, enabling scalable and intelligent defect detection.

---

# Model Architecture

**Base Model:** CLIPSeg

**Components:**

- Vision Encoder: Extracts image features
- Text Encoder: Extracts prompt features
- Segmentation Decoder: Generates pixel-wise mask

**Training Strategy:**

- Pretrained encoders frozen
- Decoder fine-tuned on drywall dataset
- Binary Cross Entropy Loss used

---

# Dataset

Dataset obtained from Roboflow in COCO format.

Contains:

- Drywall crack images (polygon annotations)
- Drywall joint images (bounding box annotations converted to masks)

All annotations converted to binary segmentation masks.

---

# Results

## Crack Segmentation

![Crack Result](results/cracks/sample_1.png)

---

## Drywall Joint Segmentation

![Drywall Result](results/drywall/sample_1.png)

---

# Performance Metrics

| Dataset | IoU | Dice Score | Pixel Accuracy |
|--------|------|-------------|----------------|
| Cracks | 0.42 | 0.56 | 0.94 |
| Drywall Joints | 0.50 | 0.65 | 0.91 |

---

# Key Features

Prompt-based segmentation  
Single model handles multiple defect types  
Vision-Language learning approach  
Custom fine-tuning pipeline  
Quantitative and qualitative evaluation  

---

# Project Structure


---

# How It Works

Input:

Image + Text Prompt

↓

Model understands prompt context

↓

Outputs segmentation mask

---

# Example

Prompt:

segment crack

Output:

Model highlights crack region

---

# Applications

Construction quality inspection  
Automated structural monitoring  
Vision-language segmentation research  
Prompt-based industrial inspection  

---

# Future Improvements

Train for more epochs to improve IoU  
Support additional defect types  
Deploy as real-time inspection tool  

---

# Acknowledgment

This project demonstrates the power of combining computer vision and natural language for intelligent defect detection.

