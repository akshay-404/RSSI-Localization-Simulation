# Wi-Fi RSSI Indoor Localization Simulation

A Python-based simulation framework for studying **Wi-Fi RSSI-based indoor localization** and evaluating the effect of RSSI noise, access-point density, and access-point placement on localization accuracy.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.2.6-orange.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3.3-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.9-orange.svg)](https://matplotlib.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.7.2-F7931E.svg)](https://scikit-learn.org/)

## Overview

Indoor localization is challenging because satellite-based positioning systems such as GNSS become unreliable inside buildings due to signal attenuation, multipath propagation, and physical obstructions. Wi-Fi RSSI provides a practical alternative because Wi-Fi access points are widely available and can provide measurable signal-strength information.

This project implements a simplified indoor Wi-Fi localization environment in Python. RSSI measurements are generated using the **log-distance path-loss model with configurable Gaussian measurement noise**. Three localization techniques are implemented and compared:

1. **Nearest Access Point (NAP)**
2. **Weighted Centroid (WC)**
3. **K-Nearest Neighbors (KNN) RSSI Fingerprinting**

The estimated position is compared with the actual user position using Euclidean localization error.

The project investigates how localization performance changes with:

* RSSI measurement noise
* Number of Wi-Fi access points
* Access-point placement
* Choice of localization algorithm

The overall objective is to provide a simple, reproducible, and extensible framework for understanding the fundamental principles and limitations of RSSI-based indoor positioning.

---

## Localization Methods

### 1. Nearest Access Point

The Nearest Access Point method assumes that the access point producing the strongest RSSI is closest to the user.

$$
i^* = \arg\max_i RSSI_i
$$

The estimated position is assigned directly to the coordinates of the selected access point.

This method is computationally simple but cannot provide continuous position estimates because every prediction is restricted to an AP location.

### 2. Weighted Centroid

The Weighted Centroid method estimates the user position as an RSSI-weighted combination of all access-point coordinates.

$$
\hat{x} =
\frac{\sum_i w_i x_i}{\sum_i w_i},
\qquad
\hat{y} =
\frac{\sum_i w_i y_i}{\sum_i w_i}
$$

where the RSSI-dependent weight is based on

$$
w_i = 10^{RSSI_i/10}.
$$

This produces a continuous position estimate while giving stronger signals greater influence.

### 3. KNN RSSI Fingerprinting

The fingerprinting approach uses an offline database containing RSSI measurements associated with known reference positions.

For a query RSSI vector, the Euclidean distance between the query and each stored fingerprint is calculated. The \(K\) most similar fingerprints are selected and their associated coordinates are used to estimate the user's position.

The baseline implementation uses:

$$
K=3.
$$

KNN fingerprinting provides a machine-learning-based baseline without requiring a complex neural-network architecture.

---

## RSSI Model

The simulation uses the log-distance path-loss model:

$$
RSSI(d) = RSSI(d_0) - 10n\log_{10}\left(\frac{d}{d_0}\right) + X_\sigma
$$

where:

* $RSSI(d_0)$ — reference RSSI
* $d_0$ — reference distance
* $n$ — path-loss exponent
* $X_\sigma$ — Gaussian RSSI noise
* $\sigma$ — noise standard deviation

The noise is modeled as:

$$
X_\sigma \sim \mathcal{N}(0,\sigma^2).
$$

This allows the simulation to reproduce increasingly variable RSSI measurements by changing $\sigma$.

---

## Baseline Configuration

| Parameter             |              Value |
| --------------------- | -----------------: |
| Indoor area           |   $20 \times 20$ m |
| Number of APs         |                  4 |
| Reference RSSI        |            −40 dBm |
| Reference distance    |                1 m |
| Path-loss exponent    |                3.0 |
| RSSI noise $\sigma$   |               2 dB |
| Test positions        |               1000 |
| KNN $K$               |                  3 |

The four access points are randomly positioned such that each AP occupies a different sub-region of the simulated environment. User positions are randomly generated throughout the $20\times20$ m area.

---

## Experiments

### Experiment 1 — Algorithm Comparison

The three localization methods are evaluated using the same simulated environment and test positions.

* Nearest AP
* Weighted Centroid
* KNN Fingerprinting

The primary metric is mean localization error, with RMSE and median error also used for evaluation.

### Experiment 2 — Effect of RSSI Noise

The RSSI noise standard deviation is varied over:

$$
\sigma \in \{0,1,2,3,4,5\}\text{ dB}.
$$

This experiment evaluates the robustness of each localization method against increasing RSSI fluctuations.

### Experiment 3 — Effect of Number of Access Points

The number of APs is varied over:

$$
N \in \{2,3,4,5,6\}.
$$

The experiment investigates whether increasing the available Wi-Fi infrastructure improves localization accuracy.

### Experiment 4 — Access-Point Placement

Different AP geometries can be evaluated to study how spatial distribution affects localization performance. This is important because two networks with the same number of APs can provide different amounts of spatial information depending on their geometry.

---

## Evaluation Metrics

The primary evaluation metric is the Euclidean localization error:

$$
e =
\sqrt{(x-\hat{x})^2+(y-\hat{y})^2}.
$$

The implementation evaluates:

* **MAE** — Mean Absolute Localization Error
* **RMSE** — Root Mean Square Error
* **Median localization error**
* **Accuracy within specified distance thresholds**
* **Cumulative Distribution Function (CDF) of localization error**

The threshold-based accuracy can be expressed as:

$$
Accuracy(r) = \frac{\\#\{e_i\leq r\}}{M}\times100.
$$

These metrics provide both an overall measure of localization accuracy and a view of the distribution of individual localization errors.

---

## Project Structure

```text
RSSI-Localization-Simulation/
|
├── data_official
│   ├── pdf
│   │   ├── ap_effect.pdf
│   │   ├── cumulative_error.pdf
│   │   ├── environment.pdf
│   │   ├── estimated_KNN (K=3).pdf
│   │   ├── estimated_Nearest AP.pdf
│   │   ├── estimated_Weighted Centroid.pdf
│   │   └── noise_effect.pdf
│   ├── ap_effect.json
│   ├── ap_effect.png
│   ├── cumulative_error.png
│   ├── environment.png
│   ├── estimated_KNN (K=3).png
│   ├── estimated_Nearest AP.png
│   ├── estimated_Weighted Centroid.png
│   ├── main.json
│   ├── noise.json
│   ├── noise_effect.png
│   └── rssi_fingerprint.csv
│
├── src
│   ├── __init.py__
│   ├── config.py
│   ├── environment.py
│   ├── localization.py
│   ├── metrics.py
│   ├── rssi.py
│   ├── simulation.py
│   └──visualization.py
│   
├── main.py
├── noise.py
├── access_point.py
└── README.md
```

The source modules separate environment generation, RSSI simulation, localization algorithms, evaluation metrics, simulation control, and visualization to keep the implementation modular and extensible.

---


## Generated Visualizations

The project generates figures including:

* Indoor localization environment and AP placement
* Actual vs. estimated positions for Nearest AP
* Actual vs. estimated positions for Weighted Centroid
* Actual vs. estimated positions for KNN
* Cumulative localization-error distribution
* Effect of RSSI noise on localization error
* Effect of the number of APs on localization error

These visualizations are used to analyze both the spatial behavior and statistical performance of the localization algorithms.

---

## Technologies Used

| Technology     | Purpose                                    |
| -------------- | ------------------------------------------ |
| Python 3.10.12 | Core implementation                        |
| NumPy          | Numerical computation and RSSI simulation  |
| Pandas         | Fingerprint and experiment data management |
| Matplotlib     | Visualization and figure generation        |
| Scikit-learn   | KNN fingerprinting                         |
| Git / GitHub   | Version control and reproducibility        |

---

## Limitations

The current implementation intentionally uses a simplified simulation model. It does not explicitly model:

* Walls and room geometry
* Multipath propagation
* Reflection, diffraction, and scattering
* Wi-Fi channel interference
* Antenna characteristics
* Frequency-dependent propagation
* Transmit-power adaptation
* Complex real-world RSSI distributions
* Multi-floor positioning

The KNN fingerprinting experiment also assumes that the training and testing environments are reasonably similar. Changes in furniture, access-point configuration, transmit power, or human occupancy can alter the RSSI distribution in a real deployment.

---

## Future Extensions

The framework can be extended to include:

* Real Wi-Fi RSSI measurements
* Weighted KNN
* Random Forest regression
* Support Vector Regression
* Gradient Boosting
* Multilayer Perceptron
* Wi-Fi Channel State Information (CSI)
* Wi-Fi Round Trip Time (RTT)
* Hybrid localization using multiple sensors

These extensions can help bridge the gap between the simplified simulation and real-world indoor positioning systems.

---

## Reproducibility

All figures generated for the project are produced using the Python simulation and visualization pipeline. The numerical results are generated programmatically rather than manually constructed. The complete implementation, simulation scripts, and project resources are maintained in this repository to facilitate reproducibility and further experimentation.

---
