# Bridge SHM Literature – Methods vs Our Multi-Sensor 1D CNN

Checked PDFs via pypdf (skimmed key sections); focus on data types, method, and similarity to our stacked multi-sensor 1D CNN for anomaly detection/forecasting.

| Paper | Data Types | Approach | Similar to ours? | Key differences |
| --- | --- | --- | --- | --- |
| Ahmad et al. 2023 – Bridge vibration energy harvesting for wireless IoT SHM | Energy harvesting context, vibration | Energy harvesting hardware review | No | Hardware/power focus; not an ML anomaly model |
| Aktan et al. 2024 – Lessons from Bridge SHM | General SHM | Lessons/review | No | Conceptual review, no model |
| Al-Ali et al. 2024 – IoT-Based Road Bridge Health Monitoring and Warning | Multi-sensor IoT (strain/accel/temp) | System design with basic analytics | No | System-level, not 1D CNN fusion |
| Al-Zuriqat et al. 2023 – Adaptive Fault Diagnosis for Simultaneous Sensor Faults | SHM sensors | Adaptive fault diagnosis (non-CNN focus) | No | Fault-detection scheme; no sliding-window multi-sensor CNN |
| Ali et al. 2025 – Advancements in energy harvesting techniques | Energy harvesting | Review | No | Power/IoT focus, no anomaly model |
| Bao et al. 2025 – Recent advances in structural health diagnosis | Various | Review | No | Survey of ML; no specific multi-sensor 1D CNN |
| Buckley et al. 2023 – Feature Extraction & Selection Benchmark for SHM | SHM features, Z24 | Classical ML feature benchmark | No | Feature selection/benchmark, not deep multi-sensor CNN |
| Bučinskas et al. 2021 – Dynamic output method, Viļnius case | Vibration | Dynamic output method | No | Modal/dynamic method, not deep CNN |
| Deng et al. 2023 – Current Development of SHM for Bridges | Various | Review | No | Broad review |
| Deraemaeker 2010 – New Trends in Vibration-Based SHM | Vibration | Review | No | Pre-deep learning, modal focus |
| Desjardins & Lau 2024 – OMA + change point detection | Vibration | OMA + change-point stats | No | Statistical/modal, not CNN |
| Eltouny et al. 2023 – Unsupervised Learning Methods for Vibration-Based SHM | Vibration | Review (unsupervised methods) | No | Survey |
| Fernandez-Navamuel et al. 2022 – Deep learning enhanced PCA for SHM | Vibration/strain/temp | DL + PCA feature reduction | Partial | Deep features but not multi-sensor sliding 1D CNN; no forecasting horizon |
| Fernandez-Navamuel et al. 2024 – DNN for damage detection (Infante Dom Henrique bridge) | Multi-sensor | DNN/CNN for detection | Partial | Multi-sensor detection only; no predictive horizon; architecture not our stacked 1D window |
| Giglioni et al. 2022 – Autoencoders for unsupervised real-time bridge health | Vibration, Z24 | Unsupervised autoencoder | No | AE reconstruction, not supervised multi-sensor CNN |
| Gomez-Cabrera & Escamilla-Ambrosio 2022 – ML techniques review | Various | Review | No | Survey |
| Gui et al. 2025 – Enhanced multi-objective optimization model | Performance indicators | Optimization model | No | Optimization, not deep CNN |
| Hasani & Freddi 2023 – Operational Modal Analysis comprehensive review | Vibration | Review | No | OMA review |
| Hoang et al. 2024 – Two-stage damage detection in Z24 (KNN + ANN) | Z24 vibration | KNN + ANN | No | Classical ML; no conv fusion or forecasting |
| Jansen & Geißler – Multi-Feature Anomaly Detection using Autoencoder | Multi-feature SHM | Autoencoder | No | AE anomaly scoring; no supervised multi-sensor CNN |
| Jiang et al. 2021 – Decentralized unsupervised diagnosis with deep auto-encoders | SHM sensors | Deep autoencoder (unsupervised) | No | AE, not supervised multi-sensor conv |
| Karakostas et al. 2024 – Seismic assessment review | Various | Review | No | Seismic SHM review |
| Lee et al. 2025 – Damage detection for railway bridges (time–frequency + cGAN) | Vibration/time–freq | cGAN/CNN on spectrograms | Partial | CNN on time–freq images; not raw stacked multi-sensor 1D; no forecasting |
| Limongelli et al. 2025 – SCSHM benchmark study | Benchmark dataset | Benchmark/description | No | Dataset/benchmark, no deep CNN method |
| Lynch & Loh 2006 – Wireless sensors review | Sensors | Review | No | Hardware/sensor network focus |
| Malekloo et al. 2022 – ML and SHM overview | Various | Review | No | Survey |
| Ni et al. 2009 – SHM system for Guangzhou TV Tower | Multi-sensor | System/technology review | No | System deployment, not CNN |
| Ni et al. 2019 – Deep learning for data anomaly detection and compression | Bridge monitoring | Deep autoencoder for compression + anomaly | Partial | Uses deep AE; not multi-sensor 1D CNN; no predictive horizon |
| Noori Hoshyar et al. 2023 – Proposed ML techniques (lab study) | Lab SHM data | Classical ML (KNN/SVM/etc.) | No | Non-CNN classical models |
| Panfeng et al. 2024 – Data repair with LSTM | SHM time series | LSTM for imputation | No | Data repair, not anomaly detection CNN |
| Pinheiro et al. 2025 – Impact of feature scaling | Various ML datasets | Methodology study | No | Feature scaling effects, no specific SHM CNN |
| Pięk & Pawłuszek-Filipiak 2025 – EGMS-PSInSAR displacement forecasting/anomaly | Remote sensing displacement | Forecasting/anomaly (non-CNN) | No | Remote sensing, not multi-sensor fused CNN |
| Providence et al. 2022 – Spatial/temporal normalization for multivariate TS prediction | Multivariate TS | Normalization + ML prediction | Partial | Multivariate prediction focus; not supervised multi-sensor CNN anomaly/forecast |
| Santaniello & Russo 2023 – Damage ID using DNN on time–frequency | Vibration → spectrograms | CNN/DNN on time–freq images | Partial | Works on transformed images; no stacked raw multi-sensor 1D; no forecasting |
| Santos-Vila et al. 2025 – Damage Detection on Real Bridges (systematic review) | Various | Review | No | Survey |
| Sarwar & Cantero 2024 – Probabilistic autoencoder-based damage assessment | Train-induced responses | Probabilistic AE | No | AE-based; not supervised multi-sensor CNN |
| Shi et al. 2025 – Decentralized damage detection for self-powered sensing | SHM sensors | Decentralized method (non-CNN) | No | Decentralized/different architecture; no fused 1D CNN |
| Soleimani-Babakamali et al. 2022 – System reliability approach (unsupervised) | SHM sensors | Unsupervised reliability modeling | No | Reliability/unsupervised; no CNN |
| Sonbul & Rashid 2023 – Wireless sensor networks for SHM (systematic study) | Sensors | Review | No | WSN focus |
| Sony et al. 2021 – Systematic review of CNN-based condition assessment | Various | Review (CNN focus) | No | Survey; no specific multi-sensor window model |
| Spencer et al. 2025 – Advances in AI for SHM (comprehensive review) | Various | Review | No | Survey |
| Sun et al. 2020 – Review of Bridge SHM aided by Big Data/AI | Various | Review | No | Survey |
| Sun et al. 2023 – Critical review for trustworthy & explainable SHM | Various | Review | No | Survey |
| Sun et al. 2023 – Predicting bridge displacement with hierarchical CNN | Loads/displacement TS | Hierarchical 1D CNN regression | Partial | CNN on time series for displacement forecast; not anomaly labels; unclear multi-sensor fusion |
| Svendsen et al. 2022 – Data-based SHM for damage detection (experimental) | Vibration (steel bridge) | Data-driven detection (classical/ML) | No | Not multi-sensor stacked 1D CNN |
| Wan et al. 2023 – Knowledge- and data-driven approaches review | Various | Review | No | Survey |
| Xu et al. 2023 – ML choices for concrete/steel bridge SHM | Various | Review | No | Survey |
| Yu et al. 2022 – Intelligent crack detection and quantification | Images | CNN on images | No | Image-based, not multi-sensor time-series |
| Zhang et al. 2022 – Ensemble learning-based forecasting (small samples) | Time series | Ensemble forecasting/classification | No | Ensemble methods; not multi-sensor CNN |
| Zhang et al. 2022 – Application of deep learning in bridge health monitoring | Various | Review | No | Survey |
| Zhang et al. 2025 – Review of SHM methods/applications for bridges | Various | Review | No | Survey |

Bottom line: none of the surveyed papers combine heterogeneous sensor types per timestep into one supervised 1D CNN for both current anomaly detection and future anomaly forecasting as we do; closest are single-task CNNs on time series (e.g., Sun 2023 displacement CNN) or general deep autoencoder approaches without multi-sensor sliding-window fusion or predictive labeling.
