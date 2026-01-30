#  FitPulse - Health Anomaly Detection System (Infosys Springboard)

An AI-powered health monitoring system designed to detect anomalies in patient vitals using Machine Learning. Developed as part of the **Infosys Springboard Internship (Batch 7)**.

##  Project Overview
**FitPulse** is a data science solution aimed at early diagnosis and preventative healthcare. By analyzing patient health data (vitals, history, and real-time metrics), the system identifies irregular patterns ("anomalies") that may indicate potential health risks.

This project was developed in **4 Strategic Milestones**, simulating a real-world software development lifecycle (SDLC).

## 📂 Project Structure & Milestones
The repository is organized into progressive milestones:

* **📁 Fitpulse_Milestone1 (Data Collection & Preprocessing):**
    * Gathering raw health datasets and handling missing values.
    * Implementing data cleaning pipelines (`fitpulse_preprocessing.py`) to ensure data quality.

* **📁 Fitpulse_Milestone2 (Feature Extraction):**
    * Analyzing correlations and selecting the most relevant health metrics.
    * Transforming raw data into meaningful features for the Machine Learning model.

* **📁 Fitpulse_Milestone3 (Anomaly Detection):**
    * Building and training the core Machine Learning models (`fitpulse_milestone2.py`).
    * Implementing algorithms to classify "Normal" vs. "Anomalous" health states.

* **📁 Fitpulse_Milestone4 (Dashboard & Deployment):**
    * Developing an interactive user interface using **Streamlit/Flask** (`app4.py`).
    * Visualizing real-time health data and displaying anomaly alerts for doctors/users.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Anomaly Detection Algorithms)
* **Web Framework:** Streamlit / Flask (for the Dashboard)
* **Visualization:** Matplotlib, Seaborn

##  How to Run
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/tanujanaidu02/Infosys_fitpulse_healthAnomali_detection.git](https://github.com/tanujanaidu02/Infosys_fitpulse_healthAnomali_detection.git)
    cd Infosys_fitpulse_healthAnomali_detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas numpy scikit-learn matplotlib streamlit
    ```

3.  **Run the Dashboard (Milestone 4)**
    To see the final output, navigate to the milestone 4 folder:
    ```bash
    cd Fitpulse_Milestone4
    python app4.py
    # OR if using Streamlit:
    # streamlit run app4.py
    ```

*Developed by  Chennamsetti Tanuja*
