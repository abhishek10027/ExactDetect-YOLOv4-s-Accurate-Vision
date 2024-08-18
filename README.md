# ExactDetect: YOLOv4's Accurate Vision

## Introduction

Object detection is a fundamental task in computer vision, involving both the identification and localization of objects within images or video frames. Unlike traditional image classification, object detection must accurately determine the positions and sizes of objects. This capability is critical for applications such as autonomous vehicles, surveillance systems, and robotics.

**ExactDetect** focuses on implementing the YOLOv4 model, a state-of-the-art method in object detection. YOLOv4 is renowned for its balance between speed and accuracy, making it ideal for real-time applications. This project explores the enhancements made in YOLOv4, particularly in detecting small and overlapping objects.

## Proposed Methodology

This project employs the YOLOv4 architecture for object detection. The key components of the proposed methodology include:

- **CSPDarkNet-53 Backbone:** Utilized for efficient feature extraction, enhancing detection accuracy for both small and large objects.
- **Path Aggregation Network (PAN):** Aggregates features across multiple levels to improve the model's ability to detect objects at different scales.
- **YOLOv4 Head:** Generates the final bounding boxes and class predictions.

The architecture of YOLOv4, as implemented in this project, is illustrated in the following diagram (Figure 1).

## Dataset

The project utilizes the **Microsoft COCO dataset**, a benchmark in computer vision with over 330,000 images across 80 object categories. This dataset is used for both training and evaluating the YOLOv4 model.

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

Four test images from the COCO dataset were used to evaluate the model. The detection results are summarized as follows:

- **Image (a):** Detected "Dog" with 94.3% confidence.

  ![image](https://github.com/user-attachments/assets/f7b1ffa2-2ee7-4763-b04b-3bf310183caa)


- **Image (b):** Detected "Person" with 98.1% confidence.
  ![image](https://github.com/user-attachments/assets/c9a28e4e-abfa-450b-a47f-683f5a1fa86a)


# The results demonstrate that the YOLOv4 model excels at detecting larger and distinct objects with high confidence.

## Conclusion

This project explores the YOLOv4 architecture for object detection, demonstrating its effectiveness in accurately detecting and localizing objects in various images. The proposed methodology leverages the strengths of YOLOv4, such as its ability to perform real-time detection with a balance of speed and accuracy. Future work may focus on enhancing the model's capability to detect small objects and adapting it to other complex object detection scenarios like instance segmentation or multi-object tracking.
