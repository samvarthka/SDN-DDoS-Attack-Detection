# Mini Project Explanation

## Abstract
This prototype demonstrates how lightweight deep learning can help identify DDoS-like patterns from Software-Defined Networking (SDN) flow statistics. A Multilayer Perceptron (MLP) is trained on generated example records and classifies each record as normal or DDoS-like. A Streamlit dashboard presents dataset summaries, model evaluation metrics, and a form for testing sample flow features.

## Existing system
Basic threshold-based monitoring can miss combinations of traffic characteristics and may require manual inspection.

## Proposed system
The proposed classroom prototype uses a lightweight neural network and multiple flow-statistics features. A simple dashboard makes the classification output understandable.

## Methodology
First, synthetic flow-statistics records are generated for normal and DDoS-like classes. The features are standardized using `StandardScaler`. The dataset is split into training and testing subsets. An MLP neural network is trained on the training subset and evaluated on the held-out subset using accuracy, precision, recall, and F1-score. Finally, the trained model is loaded by a Streamlit dashboard to classify user-entered example records.

## Limitations
This is a software prototype using synthetic data, not a live SDN deployment. It does not send network traffic or perform automatic mitigation. Synthetic test results should not be presented as real-world performance.
