# DeepSAM

Code and experiments accompanying the article **"DeepSAM"** (Neurocomputing 2026).

DeepSAM is an unsupervised anomaly detection framework for medical images. It uses a convolutional autoencoder trained only on normal samples and detects anomalies via perceptual loss at inference time.

---

## Repository structure

| Notebook | Purpose |
|----------|---------|
| [`DeepSAM_Pipeline_Main.ipynb`](#deepsam_pipeline_mainipynb) | Main pipeline — architecture, training, and AUROC / PRC evaluation |
| [`DeepSAM_Segmentation_ViT-based.ipynb`](#deepsam_segmentation_vit-basedipynb) | Segmentation comparison: ViT vs Swin-UNet vs U-Net |
| [`DeepSAM_SwinUnet_MultiDisease.ipynb`](#deepsam_swinunet_multidiseaseipynb) | Multi-disease Swin-UNet segmentation + ensemble comparison |
| [`DeepSAM_OpenMIBOOD.ipynb`](#deepsam_openmiboodipynb) | Out-of-distribution detection using OpenMIBOOD benchmark |
| [`DeepSAM_MedMNIST_C.ipynb`](#deepsam_medmnist_cipynb) | OOD robustness evaluation on corrupted MedMNIST-C data |

---

## Notebook descriptions

### `DeepSAM_Pipeline_Main.ipynb`

The **main pipeline** notebook. Start here to understand and reproduce the core DeepSAM method.

- Architecture definition (convolutional autoencoder)
- Dataset loading (brain tumor MRI, lung disease)
- Model training and evaluation
- AUROC and Precision-Recall Curve (PRC) computation for anomaly detection across multiple disease domains
- Evaluation with negative (healthy) samples

### `DeepSAM_Segmentation_ViT-based.ipynb`

Ablation study comparing **ViT-based and classical segmentation architectures** used to produce segmentation masks that feed DeepSAM.

- Vision Transformer (ViT) for semantic segmentation
- Swin-UNet (hierarchical Swin Transformer)
- Standard U-Net
- Training loops and Dice-based evaluation for each architecture

### `DeepSAM_SwinUnet_MultiDisease.ipynb`

Trains and evaluates **separate Swin-UNet segmentation models** for each disease type, then compares ensemble strategies.

- Individual models for: glioma, brain lesion, pituitary, meningioma, COVID-19 pneumonia, pneumonia, normal lung
- Ensemble methods: STAPLE, Feature/Activation Aggregation, average masks
- Head-to-head comparison of ensemble strategies on brain tumor and lung disease datasets

### `DeepSAM_OpenMIBOOD.ipynb`

Evaluates DeepSAM against the **OpenMIBOOD out-of-distribution detection benchmark**.

- Setup and dataset download for OpenMIBOOD
- Evaluation scripts for Midog, Oasis, and other domains
- Training a new autoencoder with perceptual loss
- Comparison of per-image vs. aggregated perceptual loss scores

### `DeepSAM_MedMNIST_C.ipynb`

Explores **robustness to image corruptions** using the MedMNIST-C benchmark.

- Loading clean (in-distribution) and corrupted (out-of-distribution) MedMNIST data
- Training the DeepSAM autoencoder on normal samples
- Computing perceptual loss distributions for clean vs. corrupted images
- OOD threshold estimation and detection performance at varying corruption severity levels
- Visualization of score distributions and severity curves

---

## Requirements

Dependencies and installation instructions are provided in the `## SETUP` section of each notebook. The notebooks were developed in Google Colab (GPU runtime).

Key libraries: PyTorch, torchvision, MedMNIST, medmnistc, scikit-learn, matplotlib.
