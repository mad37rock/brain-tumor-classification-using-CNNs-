# 🧠 Brain Tumor Detection using CNN and PyTorch

This project presents a robust deep learning-based solution for **brain tumor classification** using **Convolutional Neural Networks (CNN)** in **PyTorch**. By analyzing MRI images, the model assists in early and accurate diagnosis of brain tumors, achieving a remarkable **97% accuracy**.

📄 **Published Research Paper**:  
👉 [Brain Tumor Classification Using CNN on MRI Data: A PyTorch Implementation](https://www.ijisrt.com/brain-tumor-classification-using-cnn-on-mri-data-a-pytorch-implementation)

## 🚀 Project Highlights

- ✅ Achieved **97% classification accuracy** on validation data.
- 🧠 Uses a custom-built **CNN architecture** with 4 convolutional layers.
- 📈 Integrated **data augmentation** (rotation, flipping, brightness, normalization) to improve generalization.
- 🎯 Binary classification: **Tumor** vs **Healthy** MRI scans.
- ⚙️ Optimized using **Adam optimizer** with adaptive learning rate.
- 📊 Evaluated using confusion matrix, precision, recall, F1-score, and accuracy.

## 📊 Evaluation Metrics

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Healthy   | 0.98      | 0.97   | 0.98     |
| Tumor     | 0.97      | 0.98   | 0.97     |
| **Overall Accuracy** | —         | —      | **0.97**     |

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy / Matplotlib
- MRI Dataset from Kaggle

## 🎯 Objective

To develop a CNN-based system that supports medical professionals in **brain tumor detection**, reducing diagnostic errors and improving patient outcomes.

## 📎 Future Scope

- Add support for **multi-class tumor classification** (e.g., glioma, meningioma, pituitary).
- Apply **transfer learning** for performance boost.
- Integrate **Grad-CAM** for model explainability.
- Conduct **clinical validation** with real-world medical data.
