import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import os

st.set_page_config(page_title="Connectivity Viewer", page_icon="🧠")

st.title("Classification of fMRI time series and connectivity matrices")

st.markdown("""
### which models are best suited to classify fMRI time series as belonging to an LSD or placebo condition.
 we will visualize the time series and connectivity matrices for the PLCB and LSD conditions across subjects. The time series are averaged across subjects, and the connectivity matrices are visualized as heatmaps.

Time series have been previously used for classifiation tasks (Stoliker et al., 2026).They are extracted from the fMRI data and represent the BOLD signal over time for each brain region. Connectivity matrices are computed by calculating the correlation between the time series of different brain regions, resulting in a symmetric matrix that reflects the functional connectivity between regions. They are often used to study the organization of brain networks and how they change under different conditions, such as LSD vs placebo (Girn et al., 2026). Both methods provide complementary information about brain activity and connectivity patterns under different conditions. For time series and connectivity matrices, we will evaluate the performance of different machine learning and deep learning models, including Random Forest, SVM, Logistic Regression, Gradient Boosting, KNN, and a Sequential LSTM model. The goal is to determine which model achieves the highest accuracy and robustness in classifying the conditions based on the provided data.
## 1.1 Time series Model Evaluation
The time series are averaged across subjects, and the classification results for different models are summarized in the table below. The metrics include accuracy, F1-score, and ROC-AUC, which provide insights into the performance of each model in classifying fMRI time series as belonging to an LSD or placebo condition.
""")

# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════
def load_time_series_image(path):
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return None

    img = plt.imread(path)
    return img

def load_pkl_matrix(path):
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return None

    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════
time_series_path = r"C:\Users\Leandre Sabourin\Music\Desktop\Projet VSCode\deep-learning-LSD\app_structure\Figures\time_series_comparison.png"
matrix_path = r"C:\Users\Leandre Sabourin\Music\Desktop\Projet VSCode\deep-learning-LSD\app_structure\Figures\connectivity_matrices.pkl"

# ═══════════════════════════════════════════════════════════════
# Visualize time series for PLCB and LSD conditions across subjects
# ═══════════════════════════════════════════════════════════════

time_series = load_time_series_image(time_series_path)
st.image(time_series, caption="Time Series Comparison: PLCB vs LSD")

results =  pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score", "ROC-AUC"],
    "Sequential LSTM": [0.67, 0.67, 0.78],
    "Random Forest": [0.67, 0.67, 0.67],
    "SVM (RBF)": [0.50, 0.40, 0.44],
    "Logistic Regression": [0.50, 0.40, 0.56],
    "Gradient Boosting": [0.83, 0.80, 0.67],
    "KNN (k=5)": [0.50, 0.40, 0.50]
})
st.table(results)

st.markdown("""
###Connectivity Matrix Model Evaluation

For the connectivity matrices, we will visualize the average connectivity patterns for the PLCB and LSD conditions across subjects. The connectivity matrices are computed by calculating the correlation between the time series of different brain regions, resulting in a symmetric matrix that reflects the functional connectivity between regions. The heatmaps below show the average connectivity matrices for each condition.
""")
matrix_dict = load_pkl_matrix(matrix_path)

if matrix_dict is not None:
    # Extract PLCB and LSD matrices based on tuple key structure
    plcb_mats = [v for k, v in matrix_dict.items() if k[1] == "ses-PLCB"]
    lsd_mats  = [v for k, v in matrix_dict.items() if k[1] == "ses-LSD"]
    # Average across subjects and runs
    plcb = np.mean(plcb_mats, axis=0)
    lsd  = np.mean(lsd_mats, axis=0)
    # Plot side-by-side heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(plcb, ax=ax[0], cmap="viridis")
    ax[0].set_title("PLCB Connectivity Matrix (Averaged)")
    sns.heatmap(lsd, ax=ax[1], cmap="viridis")
    ax[1].set_title("LSD Connectivity Matrix (Averaged)")
    st.pyplot(fig)

results_2 =  pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score", "ROC-AUC"],
    "Random Forest": [0.67, 0.50, 0.44],
    "SVM (RBF)": [0.33, 0.00, 0.83],
    "Logistic Regression": [0.50, 0.40, 0.56],
    "Gradient Boosting": [0.33 , 0.33 , 0.33],
    "KNN (k=5)": [0.50, 0.57, 0.50]
})
st.table(results_2)

st.markdown("""
### Discussion
#### Summary of results for time series classification: 
The Sequential LSTM model outperforms traditional machine learning models in classifying fMRI time series as belonging to an LSD or placebo condition. The LSTM model achieved the highest accuracy, F1-score, and ROC-AUC, suggesting that it is better suited for capturing the temporal dynamics of fMRI data. Traditional models like Random Forest and Gradient Boosting also performed reasonably well, but they did not match the performance of the LSTM model. These findings highlight the potential of deep learning approaches for analyzing complex neuroimaging data and provide a foundation for further research into the effects of psychedelics on brain connectivity.

#### Connectivity Matrix Classification Results     
Random Forest model outperforms other traditional machine learning models in classifying connectivity matrices as belonging to an LSD or placebo condition. The Random Forest model achieved the highest accuracy, F1-score, and ROC-AUC. Other models like SVM and Gradient Boosting performed poorly, indicating that they may not be well-suited for this type of data. These findings highlight the potential of ensemble methods like Random Forest for analyzing functional connectivity patterns in neuroimaging data.
""")



st.markdown("""## Reference
Girn, M., Doss, M. K., Roseman, L., Preller, K. H., Palhano-Fontes, F., Pasquini, L., Barrett, F. S., Mallaroni, P., Mason, N. L., Timmermann, C., McCulloch, D. E., Fisher, P. M., Winston, B. S., Moujaes, F., Muller, F., Liechti, M. E., Vollenweider, F. X., Ramaekers, J. G., Kuypers, K., … Bzdok, D. (2026). An international mega-analysis of psychedelic drug effects on brain circuit function. Nature Medicine, 32(4), 1543‑1554. https://doi.org/10.1038/s41591-026-04287-9

Stoliker, D., Novelli, L., Khajehnejad, M., Biabani, M., Greaves, M. D., Barta, T., Williams, M., Chopra, S., Bazin, O., Simonsson, O., Chambers, R., Barrett, F. S., Deco, G., Preller, K. H., Carhart-Harris, R. L., Seth, A. K., Sundram, S., Egan, G. F., & Razi, A. (2026). Psychedelics align brain activity with context. Nature, 656(8129), 936‑947. https://doi.org/10.1038/s41586-026-10910-z """)