# Hematological Malignancy Classification using Advanced Deep Learning Framework

## Abstract

Hematological malignancies represent one of the most challenging areas in clinical diagnostics, requiring precise differentiation between morphologically similar cancer subtypes. This research addresses the critical need for automated classification of four major leukemia types: Chronic Myeloid Leukemia (CML), Chronic Lymphocytic Leukemia (CLL), Acute Myeloblastic Leukemia (AML), and Acute Lymphoblastic Leukemia (ALL).

Our innovative approach introduces a comprehensive framework that synergistically combines multiple state-of-the-art techniques to overcome traditional limitations in biomedical image analysis. The methodology integrates a Fully Convolutional Masked Autoencoder V2 (FCMAE V2) with a Windowed Patch Attention Transformer (WPAT) architecture, enabling sophisticated multi-scale feature extraction that captures both local cellular morphology and global tissue patterns essential for accurate leukemia classification.

To address the prevalent issue of dataset imbalance in medical imaging, we implement an Adaptive Class Distribution Balancing Generative Adversarial Network (ACDB-GAN), which intelligently generates synthetic samples while maintaining the biological authenticity of cellular structures. Additionally, our Adaptive Local Histogram Enhancement (ALHE) preprocessing technique optimizes image quality and contrast, ensuring robust feature extraction across varying imaging conditions.

Comprehensive evaluation on the challenging Raabin dataset demonstrates the superior performance of our approach, achieving an impressive 96% classification accuracy. This represents approximately a 10% improvement over existing state-of-the-art methods, establishing a new benchmark in automated leukemia diagnosis. The framework's enhanced robustness and clinical applicability make it a promising tool for supporting medical professionals in early and accurate leukemia detection.

## Key Features

- **Multi-Scale Feature Extraction**: Integration of FCMAE V2 and WPAT for comprehensive cellular pattern recognition
- **Intelligent Data Augmentation**: ACDB-GAN for addressing class imbalance while preserving biological authenticity
- **Advanced Image Preprocessing**: ALHE technique for optimal image enhancement and contrast adjustment
- **Superior Performance**: 96% classification accuracy on the Raabin dataset
- **Clinical Applicability**: Robust framework designed for real-world diagnostic scenarios

## Dataset Information

| Attribute | Description |
|-----------|-------------|
| **Dataset Name** | Raabin Hematological Malignancy Dataset |
| **Classes** | 4 (CML, CLL, AML, ALL) |
| **Image Type** | Microscopic blood cell images |
| **Format** | High-resolution digital pathology images |
| **Challenge Level** | High - morphologically similar cancer subtypes |
| **Clinical Relevance** | Direct application in leukemia diagnosis |

## Methodology

### Data Split Strategy
Our robust evaluation methodology employs a carefully designed data partitioning strategy to ensure reliable performance assessment:

- **Training Set**: 80% of the dataset
  - Primary training data: 70% of total dataset
  - Validation set: 10% of total dataset (derived from the 80% training portion)
- **Test Set**: 20% of the dataset (completely held out for final evaluation)

This configuration ensures that the model is trained on sufficient data while maintaining rigorous validation and testing protocols. The validation set is used for hyperparameter tuning and model selection during training, while the test set provides an unbiased evaluation of the final model performance.

### Training Process
1. **Preprocessing**: Application of ALHE for image enhancement
2. **Data Balancing**: ACDB-GAN generates synthetic samples for underrepresented classes
3. **Feature Learning**: FCMAE V2 learns robust feature representations through self-supervised pretraining
4. **Classification**: WPAT performs final classification with attention-based feature integration
5. **Validation**: Continuous monitoring using the 10% validation split
6. **Testing**: Final evaluation on the 20% held-out test set

## Dependencies and Libraries

### Core Deep Learning Frameworks
- **TensorFlow & Keras**: Primary framework for model implementation, providing high-level APIs for neural network construction and training
- **PyTorch**: Advanced tensor operations and dynamic computational graphs, particularly useful for research-oriented implementations
- **torchvision**: Computer vision utilities and pre-trained models for PyTorch

### Image Processing and Computer Vision
- **OpenCV (cv2)**: Comprehensive computer vision library for image manipulation, filtering, and advanced processing operations
- **PIL (Python Imaging Library)**: Basic image processing operations including filtering, enhancement, and format conversions
- **scikit-image**: Scientific image processing with specialized functions for medical imaging analysis

### Numerical Computing and Data Handling
- **NumPy**: Fundamental package for numerical computations and array operations
- **pandas**: Data manipulation and analysis, particularly useful for handling metadata and experimental results
- **pathlib**: Modern path handling for file system operations

### Evaluation and Metrics
- **PIQA (Perceptual Image Quality Assessment)**: Advanced image quality metrics including SSIM for structural similarity assessment
- **torch_fidelity**: Comprehensive evaluation metrics for generative models, ensuring synthetic data quality
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity metrics for evaluating generated images
- **scikit-learn**: Machine learning utilities including distance metrics and evaluation functions

### Visualization and Analysis
- **matplotlib**: Comprehensive plotting library for data visualization, training curves, and result analysis
- **seaborn**: Statistical data visualization built on matplotlib for enhanced plotting capabilities

### Utility Libraries
- **datetime**: Time and date handling for experiment logging and reproducibility
- **csv**: Data export and import functionality for results and metadata
- **random**: Random number generation for reproducible experiments
- **os**: Operating system interface for file and directory operations

## Performance Metrics

- **Overall Accuracy**: 96%
- **Improvement over SOTA**: ~10%
- **Robustness**: Enhanced performance across all leukemia subtypes
- **Clinical Applicability**: Validated on challenging real-world dataset

## Clinical Impact

This framework represents a significant advancement in computer-aided diagnosis for hematological malignancies. The high accuracy and robust performance make it suitable for:

- **Screening Applications**: Early detection in routine blood tests
- **Diagnostic Support**: Assisting hematologists in complex cases
- **Resource-Limited Settings**: Automated diagnosis where specialist expertise is limited
- **Research Applications**: Large-scale epidemiological studies

## Future Work

- Extension to additional hematological malignancy subtypes
- Integration with clinical decision support systems
- Real-time diagnostic applications
- Multi-modal data fusion incorporating clinical parameters

## Citation

If you use this work in your research, please cite:
```
```

## License


## Contact

