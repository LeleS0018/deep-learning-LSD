import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import os
import plotly.graph_objects as go
from nilearn import datasets, surface
import numpy as np

import plotly.io as pio

pio.renderers.default = "browser"

st.set_page_config(page_title="brain region specificity", page_icon="🧠")

st.title("Functional connectivity patterns between brain regions under LSD vs placebo")




st.markdown("""

### Does LSD alter functional connectivity patterns between brain regions compared to placebo, and whether these changes can be quantified and visualized?

Previously, it has been suggested that psychedelics decrease functional connectivity (FC) within and increase FC between most large-scale cortical networks— (Roseman et al., 2024)

Here are the steps:
- Compute connectivity matrices for LSD and placebo.
- Compare them within subjects.
- Summarize changes at edge, region, and network levels.
- Visualize them as matrices, brain maps, and graphs.
""")

network = r'C:\Users\Leandre Sabourin\Music\Desktop\Projet VSCode\deep-learning-LSD\app_structure\Figures\02-17 network heatmap.png'
glass_brain = r'C:\Users\Leandre Sabourin\Music\Desktop\Projet VSCode\deep-learning-LSD\app_structure\Figures\02-Glass_Brain.png'


import streamlit as st

col1, col2 = st.columns(2)

with col1:
    st.image(network, caption="FC changes between 17 Network", use_column_width=True)
    st.markdown(""" The Yeo 17-network is a widely used parcellation of the human brain into 17 distinct functional networks based on resting-state fMRI data. It provides a framework for understanding the organization of brain activity and connectivity patterns across different cognitive and behavioral states. The 17 networks are grouped into seven larger-scale networks, which include the visual, somatomotor, dorsal attention, ventral attention, limbic, frontoparietal, and default mode networks. """)

with col2:
    st.image(glass_brain, caption="Glass Brain", use_column_width=True)




st.image(network, caption="17-Network FC Difference (LSD – Placebo)", use_column_width=True) 
st.image(glass_brain, caption="Glass Brain Visualization of FC Changes (LSD – Placebo)", use_column_width=True)


st.markdown(""" References:
Roseman, L., Leech, R., Feilding, A., Nutt, D. J. & Carhart-Harris, R. L. The effects of psilocybin and MDMA on between-network resting state functional connectivity in healthy volunteers. Front. Human Neurosci. 8, 204 (2014).
""")