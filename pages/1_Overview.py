import streamlit as st

st.title("Overview") 
st.markdown(
    """
    This personal project aims to learn how to work with fMRI data and answer research questions related to the effects of psychedelic substances on brain activity. The project leverages two publicly available datasets from OpenNeuro:

1. Robin Carhart-Harris et al. (2020). Neural correlates of the LSD experience revealed by multimodal neuroimaging. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds003059.v1.0.0
    - 12 subjects scanned under both LSD and placebo conditions
2. Adeel Razi (2026). PsiConnect. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds006110.v1.2.2
    - 62 subjects scanned under both psilocybin for four different tasks : resting-state, music, movie and meditation
"""
)

st.title("Research Questions")


st.markdown(
    """

1. Can ML/DL models reliably classify fMRI time series and connectivity matrices as belonging to an LSD or placebo condition or between different condition (e.g., LSD vs. Psilocybin at resting state)?
2. Does LSD alter functional connectivity patterns between brain regions compared to placebo?
3. What are the differences between different tasks (e.g., resting-state vs. music-listening) in terms of functional connectivity and brain network dynamics under psilocybin?
"""
)


st.title("Key findings")
st.markdown("""
1. """)


st.title("Notes")
st.markdown(""" These results should be interpreted with caution, as it is a personal project and not a peer-reviewed study. The findings are based on the analysis of publicly available datasets and may not generalize to other populations or conditions. Further research is needed to validate these findings and explore their implications for understanding the effects of psychedelic substances on brain function.""")


st.title("References")
st.markdown("""

1. Carhart-Harris, R. L., Muthukumaraswamy, S., Roseman, L., Kaelen, M., Droog, W., Murphy, K., Tagliazucchi, E., Schenberg, E. E., Nest, T., Orban, C., Leech, R., Williams, L. T., Williams, T. M., Bolstridge, M., Sessa, B., McGonigle, J., Sereno, M. I., Nichols, D., Hellyer, P. J., … Nutt, D. J. (2016). Neural correlates of the LSD experience revealed by multimodal neuroimaging. Proceedings of the National Academy of Sciences, 113(17), 4853‑4858. https://doi.org/10.1073/pnas.1518377113

2. Girn, M., Doss, M. K., Roseman, L., Preller, K. H., Palhano-Fontes, F., Pasquini, L., Barrett, F. S., Mallaroni, P., Mason, N. L., Timmermann, C., McCulloch, D. E., Fisher, P. M., Winston, B. S., Moujaes, F., Muller, F., Liechti, M. E., Vollenweider, F. X., Ramaekers, J. G., Kuypers, K., … Bzdok, D. (2026). An international mega-analysis of psychedelic drug effects on brain circuit function. Nature Medicine, 32(4), 1543‑1554. https://doi.org/10.1038/s41591-026-04287-9
3. Novelli, L. et al. (2025). PsiConnect: A Multimodal Neuroimaging Study of Psilocybin-Induced Changes in Brain and Behaviour. https://doi.org/10.1101/2025.04.11.643415

4. Stoliker, D., Novelli, L., Khajehnejad, M., Biabani, M., Greaves, M. D., Barta, T., Williams, M., Chopra, S., Bazin, O., Simonsson, O., Chambers, R., Barrett, F. S., Deco, G., Preller, K. H., Carhart-Harris, R. L., Seth, A. K., Sundram, S., Egan, G. F., & Razi, A. (2026). Psychedelics align brain activity with context. Nature, 656(8129), 936‑947. https://doi.org/10.1038/s41586-026-10910-z """)