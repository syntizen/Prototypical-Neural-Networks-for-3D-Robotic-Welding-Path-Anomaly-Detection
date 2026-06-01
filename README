# **Metric Learning for 3D Robotic Welding Path Anomaly Detection**

An end-to-end deep metric learning framework implementing **Prototypical Networks (Few-Shot Metric Learning)** to detect structural wear and operational anomalies in industrial 6-DOF robotic welding arms. This document outlines the step-by-step mathematical, architectural, and visual pipeline of the system as it executes chronologically.

## **📌 Project Overview**

In precision manufacturing, structural wear in robot joints (e.g., backlash, motor encoder degradation, friction) leads to micro-deviations in the welding path, causing defective joints.

This repository leverages **temporal 1D-Convolutional Neural Networks** combined with **Prototypical Networks** to map multivariate end-effector trajectories into a metric space. In this learned space, normal trajectories cluster tightly around their class prototypes, while structural anomalies map far from all normal operations, allowing robust out-of-distribution (OOD) anomaly scoring using simple Euclidean metrics.

                           \+---------------------------+  
                           |  Joint-Space Trajectories |  
                           \+-------------+-------------+  
                                         |  
                                         v  (DH Forward Kinematics)  
                           \+---------------------------+  
                           |  3D Cartesian Tip Path    |  
                           \+-------------+-------------+  
                                         |  
                                         v  (Time-Series Dataset)  
                           \+---------------------------+  
                           |   1D Temporal CNN Encoder |  
                           \+-------------+-------------+  
                                         |  
                                         v  (128-D Metric Space)  
                           \+---------------------------+  
                           | Prototypical Loss Compute |  
                           \+-------------+-------------+  
                                         |  
                       \+-----------------+-----------------+  
                       |                                   |  
                       v (Normal Scenario Clustering)      v (OOD Distance Scoring)  
             \[ Prototype 1, 2, ..., 20 \]             \[ Anomaly Detection \]

## **📦 Installation & Requirements**

To set up the workspace locally and reproduce the pipeline execution, follow these steps:

1. **Clone the Repository:**  
   git clone \[https://github.com/yourusername/robotic-welding-anomaly-detection.git\](https://github.com/yourusername/robotic-welding-anomaly-detection.git)  
   cd robotic-welding-anomaly-detection

2. **Install Dependencies:**  
   Ensure Python 3.8+ and PyTorch are installed, then run:  
   pip install \-r requirements.txt

   *Required dependencies: torch, scikit-learn, matplotlib, numpy, tqdm*  
3. **Run the Code:**  
   Executing the pipeline script will automatically run the simulation, train the network, evaluate anomalies, and save all the inline diagnostic images into the assets/ folder:  
   python updated\_3d\_simulation.py

## **🛠️ Step-by-Step Simulation & Modeling Pipeline**

### **0\) Setup & Hardware Initialization**

The framework detects whether a CUDA-capable GPU is available to accelerate training, otherwise falling back gracefully to the CPU. High reproducibility is maintained by forcing fixed seed values across PyTorch, NumPy, and random state generators.

### **1\) Forward Kinematics (DH) Modeling**

We model the physical kinematics of an industrial 6-DOF manipulator using Denavit-Hartenberg (DH) parameters. The homogeneous coordinate transformation mapping Frame $i-1$ to Frame $i$ is formulated as:


$$
T_i^{i-1} = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$
By sequentially multiplying the transformations across all six joints, we compute the final end-effector (tool center point) coordinates in 3D Cartesian space:


$$T_b^0 = T_1^0 \cdot T_2^1 \cdot T_3^2 \cdot T_4^3 \cdot T_5^4 \cdot T_6^5$$

### **2\) Joint-Space Trajectory Generation**

To simulate realistic industrial operations, we generate complex, multi-turn toolpaths. Each normal scenario represents a distinct welding recipe defined in the joint space using:

* Coordinate baseline configurations spreading tasks across the spatial workspace.  
* Sinusoidal sweeps representing spatial movements.  
* Mid-trajectory shifts and drifts to simulate continuous paths.  
* Smooth, low-frequency stochastic variations to replicate active controller corrections.

### **3\) Creating the Dataset**

We synthesize 20 different normal operating scenarios (NUM\_SCENARIOS \= 20), generating 100 sample passes per scenario to compile a normal training dataset of 2,000 trajectories. Each trajectory has a sequence length of TIMESTEPS \= 128 over 3-dimensional Cartesian space coordinates $(X, Y, Z)$.

### **4\) Generating Anomalous Wear Data (Scenario 21\)**

We simulate joint wear on the physical manipulator by introducing three composite degradation modes:

1. **Joint Drift:** Slowly growing offset modeling motor encoder calibration drift or mechanical backlash.  
2. **Intermittent Jams:** Randomly freezing a joint's angle over several timesteps to mimic mechanical binding or sensory drops.  
3. **High-Frequency Jitter:** Injecting un平滑 high-frequency noise to simulate mounting bolt looseness and structural resonance.

### **5\) Trajectory Visualizations**

Below are the 3D Cartesian toolpath plots generated directly from the simulation outputs:

#### **A. Normal Operating welding Paths**

The plot below illustrates one typical 3D tip trajectory for each of the 20 normal scenarios. Notice the clean, repeatable, yet complex spatial structures modeled.

#### **B. Anomalous Wear-Simulated Path Examples**

These examples demonstrate the effects of physical mechanical wear. Minor positional drift, jitter, and abrupt geometric flattenings (caused by stuck joints) represent realistic machinery degradation.

### **6\) Few-Shot Dataset Preparation**

For deep metric optimization, trajectories are formatted as tensors of shape (Batch, Channels=3, Timesteps=128) and fed to an episodic sampler.

### **7\) 1D Temporal CNN Encoder**

The network maps the multivariate time-series curves into a feature space using a deep 1D-CNN architecture:

Input: (3, 128\)   
  └── 1D Conv (Kernel=7, Output=32, Padding=3) \-\> BatchNorm \-\> ReLU  
        └── 1D Conv (Kernel=5, Output=64, Padding=2) \-\> BatchNorm \-\> ReLU  
              └── 1D Conv (Kernel=3, Output=128, Padding=1) \-\> BatchNorm \-\> ReLU  
                    └── Adaptive Average Pooling (Compresses time dimension to 1\)  
                          └── Flatten \-\> Linear Layer \-\> Output Embedding: (128-D Space)

### **8\) Episodic Prototypical Training**

During episodic metric training, the network is constrained to optimize relative spatial distances. Within each training episode (8-way 5-shot), the class prototypes $c_n$ are computed as the support set mean embeddings:


$$c_n = \frac{1}{|S_n|} \sum_{x \in S_n} f_\phi(x)$$
The probability distribution for a query sample $q$ across all classes is optimized using Euclidean metrics in the latent space:


$$p(y = n \mid q) = \frac{\exp(-\|f_\phi(q) - c_n\|_2^2)}{\sum_{n'} \exp(-\|f_\phi(q) - c_{n'}\|_2^2)}$$

### **9\) Final Prototype Construction**

After metric optimization completes, the encoder maps all 2,000 normal database samples. Their averages define the 20 official, static class prototypes in the 128D embedding space.

### **10\) Anomaly Distance Scoring**

When testing a query trajectory $x_{\text{test}}$, its anomaly score $A(x_{\text{test}})$ is the Euclidean distance to the nearest normal prototype:


$$A(x_{\text{test}}) = \min_{n \in \{1,\ldots,20\}} \| f_\phi(x_{\text{test}}) - c_n \|_2$$

### **11\) Performance Evaluation & Spatial Analysis**

The outputs below confirm the performance of the Prototypical Network at classifying operational health:

#### **A. Anomaly Detection Performance**

* **Anomaly Score Histogram (Left):** Normal validation samples (blue) occupy small distance metrics. Wear anomalies (orange) map to distant, distinct metric spaces, showing minimal overlap.  
* **ROC Curve (Right):** The system achieves outstanding classification performance with a high Area Under the Curve value ($\text{AUC} \approx 0.97$).

#### **B. Embedding Space Cluster Map (3D t-SNE)**

Reducing the 128D latent representation space to 3D highlights the geometric partitioning learned by the encoder:

* **Colored Spheres (Scenarios 0-19):** Normal operational profiles form dense, well-segregated clusters.  
* **Black Crosses ('x'):** Anomalous trajectories fall far outside the boundary of any normal task clusters, indicating clear out-of-distribution detection.

## **🗃️ Repository Structure**

├── updated\_3d\_simulation.py   \# Unified 6-DOF simulation, training, & saving pipeline  
├── README.md                  \# Project documentation (You are here)  
├── requirements.txt           \# Python dependency requirements  
├── assets/                    \# Auto-saved visualization outputs (Git-tracked)  
│   ├── one\_per\_scenario.png   \# 3D Normal scenario paths  
│   ├── anomaly\_examples.png   \# 3D Simulated wear paths  
│   ├── anomaly\_evaluation.png \# Evaluation metrics (ROC / Histogram)  
│   └── tsne\_3d.png            \# Latent space 3D t-SNE plot  
└── data/                      \# Auto-saved compressed numpy datasets  
    ├── weld\_synth\_train.npz  
    ├── weld\_synth\_test\_anom.npz  
    └── weld\_prototypes.npy

## **📄 License**

This project is licensed under the MIT License \- see the [LICENSE](http://docs.google.com/LICENSE) file for details.














