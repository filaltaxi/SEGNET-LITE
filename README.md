

# 🧠 SegNet-Lite: Hardware-Accelerated Prostate Lesion Segmentation

🚀 Efficient, privacy-preserving, embedded AI prototype for **clinically significant prostate cancer (csPCa)** detection from multi-parametric MRI.

> **New to this?** Read [SegNet-Lite Explained in Plain Language](EXPLAINED.md): how it works, what the numbers mean, and its current limits, with no jargon.
>
> **Status:** research prototype. It has not been clinically validated and is not a medical device.

Developed by **Team GenHacks**
B.Tech Electronics & Communication Engineering – RSET

**Team Members:**

* ARUN K
* OMAR SHERIFF
* ADARSH NAIR

**Problem Provider:** Dr. Ajith Toms – Senior Consultant Radiology, Rajagiri Hospital
**Project Mentor:** Ms. Preethi Bhaskaran – Assistant Professor, Dept. of ECE, RSET

---

# 📌 Project Overview

SegNet-Lite is a **two-stage cascaded deep learning system** designed for:

1️⃣ Prostate gland segmentation
2️⃣ Lesion segmentation inside the prostate ROI

Unlike conventional cloud-based AI systems, this solution runs entirely on an **Embedded Intelligent Node (ZCU104 FPGA)** for:

* ✅ 100% patient data privacy
* ✅ Low latency
* ✅ Cost-effective deployment
* ✅ Real-time inference

---

# 🏥 Clinical Motivation

### Limitations of Existing AI Systems

* ❌ Require expensive GPU server racks
* ❌ Cloud-based processing risks patient privacy
* ❌ High false positives from whole-pelvis analysis
* ❌ Not suitable for real-time clinical use

### Our Clinical Advantage

* ✔ Cascaded ROI-focused segmentation
* ✔ INT8 quantized model (75% memory reduction)
* ✔ Fully on-device inference
* ✔ 93%+ precision in lesion detection (prototype result, measured on training data)
* ✔ Aims to reduce unnecessary biopsies

---

# 🗂 Dataset

### PICAI Dataset

* 1500 bpMRI scans
* 1476 patients
* 425 clinically significant PCa cases
* 220 cases with expert-annotated lesion masks

Each case includes:

* T2W MRI
* ADC MRI
* HBV MRI

---

# 🧩 System Architecture

## 🔷 Stage 1: Prostate Gland Segmentation

* Multi-modal input (ADC, HBV, T2W)
* SegNet-Lite encoder-decoder
* Output: Binary prostate mask

## 🔷 Stage 2: Lesion Segmentation

* ROI crop from gland mask
* Patch-based lesion detection
* Hybrid Dice + Focal Loss optimization

---

# 🏗 Model Architecture

Custom lightweight SegNet variant:

* Encoder-Decoder structure
* Conv → BatchNorm → ReLU blocks
* Nearest neighbor upsampling
* Raw logits output (DPU compatible)

See training implementation in:
📄 `segnet.py` 

---

# ⚙️ Training Pipeline

* Input resized to 192×192
* Patch size: 64×64
* Z-score normalization
* Dice + Focal Loss
* Adam optimizer
* 70 epochs training

Loss Function:

* Dice Loss (handles class imbalance)
* Focal Loss (focuses on tiny lesions)

---

# 🚀 Hardware Deployment

## Embedded Intelligent Node

**Board:** Xilinx ZCU104
**Acceleration:** DPU via Vitis AI
**Quantization:** INT8

### Software Flow

* Image loading
* Preprocessing
* Mask generation
* Cascaded inference

### Hardware Flow

* Model compressed to INT8
* Executed on DPU
* Real-time inference
* Zero cloud dependency

See full hardware inference pipeline in:
📄 `pipeline.py` 

---

# 📊 Quantitative Results

### Prostate Segmentation

* Dice Score
* IoU
* Precision
* Recall
* Accuracy

### Lesion Segmentation

* Dice
* Precision ≈ 93–95%
* Reduced false positives
* ROI-focused accuracy boost

---

# 🧪 Performance Metrics (Embedded)

* Prostate DPU inference time (ms)
* Lesion DPU inference time (ms)
* Total latency
* FPS
* Throughput (samples/sec)

All performance printed in runtime summary.

---

# 📂 Project Structure

```
SEGNET-LITE/
│
├── segnet.py              # Training + model definition
├── pipeline.py            # DPU inference pipeline
├── segnet_lite.xmodel     # Quantized prostate model
├── segnet_lite_lesion.xmodel
├── segnet_lite_lesion.pth
└── README.md
```

---

# 🔬 Prototype Evolution

## Alpha Prototype (Software Only)

* Standard SegNet
* FP32 weights
* Binary Cross Entropy Loss
* High latency
* Many false positives

## Beta Prototype (Final System)

* Cascaded architecture
* Dice + Focal Loss
* INT8 quantization
* FPGA deployment
* Embedded Intelligent Node
* Working prototype (not yet clinically validated)

---

# 🔒 Privacy & Security

* No cloud processing
* No data transmission
* 100% on-device inference
* Designed to keep patient data inside the hospital

---

# 🔮 Future Roadmap

### Technical

* Doctor-friendly UI dashboard
* PACS/DICOM integration
* Versal AI Edge board migration

### Clinical

* Multi-center dataset expansion
* Radiologist validation
* Regulatory pathway planning

---

# 📚 References

* Ronneberger et al., U-Net, MICCAI 2015
* Lin et al., Focal Loss, ICCV 2017
* Wang et al., Cascade SegNet, Medical Image Analysis
* Qiu et al., Embedded FPGA for CNN

---

# 🏆 Final Outcome

✔ Solved tiny lesion imbalance problem
✔ Reached about 93–95% lesion precision in prototype testing
✔ Successfully deployed medical AI on FPGA
✔ Showed that a real-time embedded diagnostic aid is feasible

---

# 🤝 Support Needed

* Clinical validation partnerships
* Regulatory mentorship
* Funding for next-gen FPGA hardware

---

# 📌 How to Run (Training)

```bash
python segnet.py
```

# 📌 How to Run (Hardware Inference on ZCU104)

```bash
python pipeline.py
```

---

# 📢 Conclusion

SegNet-Lite turns a theoretical medical AI model into a:

* Practical
* Secure
* Low-cost
* Real-time
* Promising prototype of an embedded diagnostic aid

