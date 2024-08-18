# ExactDetect: YOLOv4's Accurate Vision

## Introduction

**ExactDetect: YOLOv4's Accurate Vision** enhances object detection by combining object categorization and localization in a single model. Leveraging YOLOv4 with the CSPDarkNet-53 backbone, this project improves detection accuracy and speed compared to traditional ResNet-based architectures. Trained on the COCO.NAMES dataset, which includes over 330,000 images across 80 categories, **ExactDetect** delivers superior performance, particularly in detecting larger objects with high confidence. This makes it an ideal solution for real-time applications requiring precise object detection.

## Proposed Methodology

This project employs the YOLOv4 architecture for object detection. The key components of the proposed methodology include:

- **CSPDarkNet-53 Backbone:** Utilized for efficient feature extraction, enhancing detection accuracy for both small and large objects.
- **Path Aggregation Network (PAN):** Aggregates features across multiple levels to improve the model's ability to detect objects at different scales.
- **YOLOv4 Head:** Generates the final bounding boxes and class predictions.

The architecture of YOLOv4, as implemented in this project, is illustrated in the following diagram (Figure).
![image](https://github.com/user-attachments/assets/a105f287-5ff0-4ea4-a25b-b7e11efe6f0d)


## Dataset

For this project, we utilized the **COCO.NAMES** dataset, a comprehensive resource widely recognized in the field of computer vision for image classification and object detection tasks. The COCO.NAMES dataset is designed for large-scale object recognition, segmentation, and labeling, featuring over 330,000 images annotated with 80 different object categories. Each image is also accompanied by five captions that describe the scene, providing rich contextual information for training models.

The extensive diversity and detailed annotations of the COCO.NAMES dataset make it an essential tool for developing and evaluating state-of-the-art object recognition and segmentation models. In our project, this dataset was pivotal for training the neural network, enabling precise identification of object categories and improving the overall accuracy of our detection system.

## System Requirements

### Hardware Configuration

| Item               | Specifications   |
|--------------------|------------------|
| Operating System   | Windows 10       |
| CPU                | Intel Core i5    |
| GPU                | NVIDIA GTX 1050  |
| RAM                | 16 GB            |
| Storage            | 512 GB SSD       |

### Software Configuration

| Item               | Specifications                    |
|--------------------|-----------------------------------|
| Programming Language | Python 3.8                        |
| Deep Learning Framework | TensorFlow 2.x or PyTorch       |
| Libraries          | OpenCV, NumPy, Matplotlib, YOLOv4   |

## Training Details

The YOLOv4 model is fine-tuned on the COCO dataset. The training process includes:

- **Data Augmentation:** Techniques like random cropping, flipping, and color adjustment are applied to increase the diversity of the training data.
- **Batch Normalization:** Improves training stability and speeds up convergence.
- **Learning Rate Scheduling:** Dynamically adjusts the learning rate during training to optimize model performance.

## Testing Details

During testing, images are resized and passed through the YOLOv4 model. A confidence threshold of 0.5 is used to filter out low-confidence predictions. The results include:

- Bounding boxes with object labels.
- Confidence scores for each detected object.

## Results Description

The detection results for randomly selected images from the COCO.NAMES test set are summarized below

- **Figure(a):** The model successfully detected a dog with a confidence score of 70.43% and a cat with a confidence score of 94.39%.

  ![image](https://github.com/user-attachments/assets/f7b1ffa2-2ee7-4763-b04b-3bf310183caa)


- **Figure (b):** Detected objects include a person with a high confidence score of 99.42%, a dog at 98.50%, and a horse with 77.92%
  ![image](https://github.com/user-attachments/assets/c9a28e4e-abfa-450b-a47f-683f5a1fa86a)


# The results demonstrate that the YOLOv4 model excels at detecting larger and distinct objects with high confidence.

## Conclusion

**ExactDetect: YOLOv4's Accurate Vision** demonstrates significant advancements in object detection by effectively integrating object categorization and localization within complex scenes. Leveraging the power of Deep Neural Networks (DNNs), our model showcases superior performance compared to traditional methods. The implementation of Scaled YOLOv4, which utilizes optimal network scaling strategies, results in the YOLOv4-CSP-P5-P6-P7-P8 networks. These networks, powered by the CSPDarkNet-53 backbone, not only achieve higher accuracy in object detection but also enhance categorization performance.

Experimental results validate that our YOLOv4 lite model outperforms state-of-the-art techniques, providing a more efficient and accurate solution for real-time object detection. This positions ExactDetect as a leading approach in the field, combining speed, accuracy, and robustness for various applications in computer vision.
