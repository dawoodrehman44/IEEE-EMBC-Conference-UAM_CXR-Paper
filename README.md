# UAM-CXR: Uncertainty-Aware Multimodal Learning for Chest X-ray Classification via Cross-Modal Conformal Prediction

<p align="center">
  <img src="UAM_CXR_Architecture_Plot.png" alt="UAM-CXR Framework Architecture" width="1000"/>
</p>
---

## 🔥 Highlights
- **First end-to-end trainable framework** combining focal loss, class weighting, and learnable conformal prediction for medical imaging.
- **Cross-modal conformal scoring** using vision + text features (unlike vision-only baselines).
- **Mathematically guarantee 90% coverage** with efficient 2-3 disease prediction sets. 
- **Comprehensive uncertainty quantification:** Aleatoric + Epistemic + Calibration + Conformal. 
- **State-of-the-art performance:** AUC 0.92-0.96, outperforming strong baselines. 
- **Joint Optimaztion** Classification, contrastive alignment, uncertainty estimation, and conformal scoring trained together.  

---

## 📋 Abstract
### Problem 

- Existing models lack uncertainty quanitification for safe clinical deployment.  
- No mathematical guarantees on prediction reliability.

**Key Innovation**:
1. **Uncertainty-Aware Classification with Focal Loss** 
2. **Contrastive Vision-text Alignment** 
3. **Learnable Conformal Prediction** 
4. **End-to-end Joint Optimization**

---

## 🏗️ Architecture Overview

Our framework consists of these critical components:

1. **Dual Encoding**: Vision Encoder + Text Encoder (Both Trainable).
2. **Contrastive Alignment**: Align vision and text in shared semantic space with InfoNCE loss. 
3. **Cross-Modal Fusion**: Aligned features v and t.
4. **Dual-Head Prediction**: Shared backbone, Classification head, Uncertainty Head (All trainable).
5. **Learnable Conformal Scoring**: Learn to identify unreliable predictions using all modalities.

---

## 📁 Repository Structure
```bash
├── Configuration/
│   └── configuration.json           
├── Data_Processing/
│   └── data_processing.py              
├── Evaluation_Metrics/
│   └── evaluation_metrics.py   
├── Experiments/
│   ├── Comparative_Case_Study_Analysis.py
│   └── Uncertainty_Analysis.py
├── Figures/
│   └── UAM_CXR_Architecture_Plot.pdf
├── Model/
│   ├── loss_function.py
│   └── model.py                  
├── Visualization/
│   ├── metrics_tracker.py                
│   ├── calibration.py                     
│   └── variational_linear.py           
├── main.py
├── train.py
├── valid.py                                 
├── requirements.text                     
└── README.md                              


```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/dawoodrehman44/IEEE-EMBC-Conference-UAM_CXR-Paper.git
cd IEEE_EMBC_2026

```
### Create environment
```bash
conda create -n UAM_CXR_med python=3.8
conda activate UAM_CXR_med

# Install dependencies
pip install -r requirements.txt
```

## Training
### Train the UAM_CXR Framework
```bash
python main.py \
    --model train \
    --config configuration/configuration.json \
    --data_path /path/to/training \

```

## Testing
### Perform comprehensive uncertainty and conforaml predeiction analysis
```bash
python Experiments/case_study_visualization.py \
    --Experiments/ablation.py \
    --data_path /path/to/validation \
    --mc_samples 1000
```

## 🤝 Acknowledgments
We thank the creators of MIMIC-CXR and IU-Xray datasets and all the models used in this work, for making them publicly available to the community.

## Contact
For questions or collaborations, please contact: 
Dawood Rehman – [dawoodrehman1297@gapp.nthu.edu.tw]