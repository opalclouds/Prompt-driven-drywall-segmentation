<div align="center">

<img src="https://img.shields.io/badge/Vision--Language-CLIPSeg-blueviolet?style=for-the-badge&logo=pytorch" />
<img src="https://img.shields.io/badge/Task-Image%20Segmentation-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Domain-Construction%20AI-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Model-Fine--Tuned-red?style=for-the-badge" />

# 🧱 Prompt-Driven Drywall Defect Segmentation

### *Identify construction defects with nothing but a sentence.*

**"segment crack"** → model highlights the crack.  
**"segment drywall joint"** → model highlights the joint.  
One model. Any defect. Zero retraining.

</div>

---

## ✨ What Is This?

Traditional defect detection pipelines require training a **separate model for each defect class** : cracks, joints, holes, and so on. That's slow, expensive, and rigid.

This project flips that paradigm. By fine-tuning **CLIPSeg**, a vision-language segmentation model on drywall imagery, we enable **natural language prompts to control what gets segmented**. Want to look for cracks? Type it. Joints? Type that instead. The same model handles both.

This makes the system **flexible, scalable, and inspection-ready** for real-world construction workflows.

---

## 🎯 Results

### 🔴 Crack Segmentation
> Prompt: `"segment crack"`

<div align="center">
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/crack_sample_1.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/crack_sample_2.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/crack_sample_3.png" width="280"/></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/crack_sample_4.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/crack_sample_5.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/crack_sample_6.png" width="280"/></td>
  </tr>
</table>
<sub><i>The model precisely isolates crack regions from complex drywall textures.</i></sub>
</div>

---

### 🟡 Drywall Joint Segmentation
> Prompt: `"segment drywall joint"`

<div align="center">
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/drywall_sample_1.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/drywall_sample_2.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/drywall_sample_3.png" width="280"/></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/drywall_sample_4.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/drywall_sample_5.png" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/opalclouds/Prompt-driven-drywall-segmentation/main/results/drywall_sample_6.png" width="280"/></td>
  </tr>
</table>
<sub><i>Panel joints are accurately delineated even in areas with low visual contrast.</i></sub>
</div>

---

## 📊 Performance Metrics

<div align="center">

| Defect Class | IoU ↑ | Dice Score ↑ | Pixel Accuracy ↑ |
|:---:|:---:|:---:|:---:|
| 🔴 Cracks | `0.42` | `0.56` | `0.94` |
| 🟡 Drywall Joints | `0.50` | `0.65` | `0.91` |

</div>

> **Note:** High pixel accuracy reflects the model's ability to correctly classify background regions. IoU and Dice scores are moderate at current training epochs — further fine-tuning is expected to improve mask precision significantly.

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────┐
│                      CLIPSeg                         │
│                                                      │
│   📷 Image  ──► Vision Encoder (frozen) ──►  ┐       │
│                                               ├──► Decoder ──► 🎭 Mask
│   📝 Prompt ──► Text Encoder  (frozen) ──►  ┘       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

| Component | Status | Role |
|---|:---:|---|
| **Vision Encoder** | ❄️ Frozen | Extracts rich spatial features from the input image |
| **Text Encoder** | ❄️ Frozen | Encodes the natural language prompt into a feature vector |
| **Segmentation Decoder** | 🔥 Fine-tuned | Fuses both modalities → outputs a pixel-wise binary mask |

**Training Details:**
- Pretrained CLIP encoders are **frozen** to preserve rich vision-language representations
- Only the **decoder is fine-tuned** on the drywall dataset
- Loss function: **Binary Cross Entropy**

---

## 📁 Dataset

- **Source:** [Roboflow](https://roboflow.com/) — COCO format
- **Crack images:** Polygon annotations → converted to binary masks
- **Joint images:** Bounding box annotations → converted to binary masks
- **Task type:** Binary segmentation (defect vs. background)

---

## 🚀 How It Works
```python
# 1. Load image + write your prompt
image = load_image("drywall_sample.jpg")
prompt = "segment crack"

# 2. Run inference
mask = clipseg_model(image, prompt)

# 3. Visualize
overlay_mask_on_image(image, mask)
```

The model understands the **semantic meaning** of your prompt and localizes the corresponding region — no class IDs, no label mappings, just plain English.

---

## 📂 Project Structure
```
Prompt-driven-drywall-segmentation/
│
├── 📓 Prompted_Segmentation_clean.ipynb   # Full training & inference notebook
├── 📁 results/
│   ├── crack_sample_1.png
│   ├── crack_sample_2.png
│   ├── crack_sample_3.png
│   ├── crack_sample_4.png
│   ├── crack_sample_5.png
│   ├── crack_sample_6.png
│   ├── drywall_sample_1.png
│   ├── drywall_sample_2.png
│   └── ...
└── 📄 README.md
```

---

## 💡 Key Features

- 🔤 **Prompt-based control** — change what you detect by changing your text
- 🧠 **Single model, multiple classes** — no retraining needed for new defect types
- ⚡ **Vision-Language fusion** — powered by CLIP's cross-modal understanding
- 🎯 **Custom fine-tuning pipeline** — tailored for construction domain imagery
- 📐 **Quantitative & qualitative evaluation** — IoU, Dice, Pixel Accuracy + visual outputs

---

## 🔮 Future Improvements

- [ ] Train for more epochs to push IoU beyond `0.60`
- [ ] Expand to additional defect types: holes, water damage, peeling paint
- [ ] Package as a **REST API** for real-time site inspection tools
- [ ] Integrate with **mobile/drone camera feeds** for on-site deployment
- [ ] Experiment with **SAM (Segment Anything Model)** as an alternative backbone

---

## 🛠️ Getting Started
```bash
# Clone the repo
git clone https://github.com/opalclouds/Prompt-driven-drywall-segmentation.git
cd Prompt-driven-drywall-segmentation

# Install dependencies
pip install transformers torch torchvision pillow matplotlib

# Open the notebook
jupyter notebook "Copy_of_Prompted_Segmentation (1).ipynb"
```

---

## 🙏 Acknowledgements

- [CLIPSeg — Lüddecke & Ecker, 2022](https://arxiv.org/abs/2112.10003) for the vision-language segmentation backbone
- [Roboflow](https://roboflow.com/) for dataset hosting and annotation tools
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for the CLIPSeg implementation

---

<div align="center">

*Built to show that computer vision and natural language can work together for smarter, more flexible industrial inspection.*

**⭐ Star this repo if you found it useful!**

</div>
