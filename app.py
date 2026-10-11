import streamlit as st
import os
import nibabel as nib
import nilearn as nl
from nilearn import plotting
from PIL import Image
import plotly.graph_objects as go
from nilearn import datasets, surface
import numpy as np
from nilearn import plotting, image
import nibabel as nib
import plotly.io as pio


# --- Sidebar styling ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 180px;
            min-width: 180px;
            background-color: #f9fafb;
            padding: 0.4rem 0.2rem;
        }

        [data-testid="stSidebar"] * {
            font-size: 2rem;
            line-height: 2.2rem;
        }

        .sidebar-section-title {
            font-size: 2.25rem;
            font-weight: 600;
            color: #374151;
            margin-top: 0.8rem;
            margin-bottom: 0.2rem;
        }

        .sidebar-link {
            font-size: 1.25rem;
            color: #2563eb;
            text-decoration: none;
        }

        .sidebar-link:hover {
            color: #1d4ed8;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar content ---
with st.sidebar:

    st.markdown("<div class='sidebar-section-title'>Datasets</div>", unsafe_allow_html=True)
    st.markdown("""
        <a class='sidebar-link' href='https://openneuro.org/datasets/ds003059'>1. ds003059 (LSD) </a><br>
         <a class='sidebar-link' href='https://openneuro.org/datasets/ds006110/versions/1.2.2'>2. ds006110 (Psilocybin)</a><br>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section-title'>Project link</div>", unsafe_allow_html=True)
    st.markdown("""
        <a class='sidebar-link' href='https://linkedin.com/in/yourprofile'>LinkedIn</a><br>
    """, unsafe_allow_html=True)


    st.markdown("<div class='sidebar-section-title'>Contact me</div>", unsafe_allow_html=True)
    st.markdown("""
        <a class='sidebar-link' href='https://linkedin.com/in/yourprofile'>LinkedIn</a><br>
        <a class='sidebar-link' href='https://github.com/yourhandle'>GitHub</a>
    """, unsafe_allow_html=True)


# ------------- Root files --------------- #
root = r"E:\fMRI dataset\LSD-PLACEBO"

# ---------------------------------------- #

st.set_page_config(
    page_title="Psychedelic Neuroimaging Project",
    page_icon="👋",
)

st.title("Psychedelic Neuroimaging Project")




#_____ plotly renderer for interactive 3D brain visualization
pio.renderers.default = "browser"   # ensures Plotly works everywhere
atlas_path = r"C:\Users\Leandre Sabourin\nilearn_data\schaefer_2018\Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

# Load FC matrices for each subject
base_dir = r"C:\Users\Leandre Sabourin\Music\Desktop\Projet VSCode\deep-learning-LSD\time_series"
subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("sub-")])
conditions = ["PLCB", "LSD"]

fc = {sub: {} for sub in subjects}

for sub in subjects:
    for cond in conditions:
        path = os.path.join(base_dir, sub, cond, "time_series.npy")
        ts = np.load(path)
        fc_mat = np.corrcoef(ts.T)
        fc[sub][cond] = fc_mat

# Fisher z-transform
def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

z_plcb = np.stack([fisher_z(fc[sub]["PLCB"]) for sub in subjects])
z_lsd  = np.stack([fisher_z(fc[sub]["LSD"])  for sub in subjects])

# ROI-level FC difference
mean_delta_roi = (z_lsd - z_plcb).mean(axis=0)


roi_labels = np.array(['Background', '7Networks_LH_Vis_1', '7Networks_LH_Vis_2', '7Networks_LH_Vis_3', '7Networks_LH_Vis_4', '7Networks_LH_Vis_5', '7Networks_LH_Vis_6', '7Networks_LH_Vis_7', '7Networks_LH_Vis_8', '7Networks_LH_Vis_9', '7Networks_LH_SomMot_1', '7Networks_LH_SomMot_2', '7Networks_LH_SomMot_3', '7Networks_LH_SomMot_4', '7Networks_LH_SomMot_5', '7Networks_LH_SomMot_6', '7Networks_LH_DorsAttn_Post_1', '7Networks_LH_DorsAttn_Post_2', '7Networks_LH_DorsAttn_Post_3', '7Networks_LH_DorsAttn_Post_4', '7Networks_LH_DorsAttn_Post_5', '7Networks_LH_DorsAttn_Post_6', '7Networks_LH_DorsAttn_PrCv_1', '7Networks_LH_DorsAttn_FEF_1', '7Networks_LH_SalVentAttn_ParOper_1', '7Networks_LH_SalVentAttn_FrOperIns_1', '7Networks_LH_SalVentAttn_FrOperIns_2', '7Networks_LH_SalVentAttn_PFCl_1', '7Networks_LH_SalVentAttn_Med_1', '7Networks_LH_SalVentAttn_Med_2', '7Networks_LH_SalVentAttn_Med_3', '7Networks_LH_Limbic_OFC_1', '7Networks_LH_Limbic_TempPole_1', '7Networks_LH_Limbic_TempPole_2', '7Networks_LH_Cont_Par_1', '7Networks_LH_Cont_PFCl_1', '7Networks_LH_Cont_pCun_1', '7Networks_LH_Cont_Cing_1', '7Networks_LH_Default_Temp_1', '7Networks_LH_Default_Temp_2', '7Networks_LH_Default_Par_1', '7Networks_LH_Default_Par_2', '7Networks_LH_Default_PFC_1', '7Networks_LH_Default_PFC_2', '7Networks_LH_Default_PFC_3', '7Networks_LH_Default_PFC_4', '7Networks_LH_Default_PFC_5', '7Networks_LH_Default_PFC_6', '7Networks_LH_Default_PFC_7', '7Networks_LH_Default_pCunPCC_1', '7Networks_LH_Default_pCunPCC_2', '7Networks_RH_Vis_1', '7Networks_RH_Vis_2', '7Networks_RH_Vis_3', '7Networks_RH_Vis_4', '7Networks_RH_Vis_5', '7Networks_RH_Vis_6', '7Networks_RH_Vis_7', '7Networks_RH_Vis_8', '7Networks_RH_SomMot_1', '7Networks_RH_SomMot_2', '7Networks_RH_SomMot_3', '7Networks_RH_SomMot_4', '7Networks_RH_SomMot_5', '7Networks_RH_SomMot_6', '7Networks_RH_SomMot_7', '7Networks_RH_SomMot_8', '7Networks_RH_DorsAttn_Post_1', '7Networks_RH_DorsAttn_Post_2', '7Networks_RH_DorsAttn_Post_3', '7Networks_RH_DorsAttn_Post_4', '7Networks_RH_DorsAttn_Post_5', '7Networks_RH_DorsAttn_PrCv_1', '7Networks_RH_DorsAttn_FEF_1', '7Networks_RH_SalVentAttn_TempOccPar_1', '7Networks_RH_SalVentAttn_TempOccPar_2', '7Networks_RH_SalVentAttn_FrOperIns_1', '7Networks_RH_SalVentAttn_Med_1', '7Networks_RH_SalVentAttn_Med_2', '7Networks_RH_Limbic_OFC_1', '7Networks_RH_Limbic_TempPole_1', '7Networks_RH_Cont_Par_1', '7Networks_RH_Cont_Par_2', '7Networks_RH_Cont_PFCl_1', '7Networks_RH_Cont_PFCl_2', '7Networks_RH_Cont_PFCl_3', '7Networks_RH_Cont_PFCl_4', '7Networks_RH_Cont_Cing_1', '7Networks_RH_Cont_PFCmp_1', '7Networks_RH_Cont_pCun_1', '7Networks_RH_Default_Par_1', '7Networks_RH_Default_Temp_1', '7Networks_RH_Default_Temp_2', '7Networks_RH_Default_Temp_3', '7Networks_RH_Default_PFCv_1', '7Networks_RH_Default_PFCv_2', '7Networks_RH_Default_PFCdPFCm_1', '7Networks_RH_Default_PFCdPFCm_2', '7Networks_RH_Default_PFCdPFCm_3', '7Networks_RH_Default_pCunPCC_1', '7Networks_RH_Default_pCunPCC_2'])
roi_values = mean_delta_roi.mean(axis=1)  # average across connections
new_data = np.zeros_like(atlas_data)

for i, lbl in enumerate(roi_labels):
    if lbl != "Background":
        new_data[atlas_data == (i+1)] = roi_values[i]

new_img = nib.Nifti1Image(new_data, atlas_img.affine)

pio.renderers.default = "browser"   # ensures Plotly works everywhere

st.header("Interactive 3D Brain (LSD – Placebo)")

# Load fsaverage surface
fsaverage = datasets.fetch_surf_fsaverage()

# Load mesh geometry
coords_left, faces_left = surface.load_surf_mesh(fsaverage.pial_left)
coords_right, faces_right = surface.load_surf_mesh(fsaverage.pial_right)

# Convert your ROI-level FC change image to surface texture
texture_left = surface.vol_to_surf(new_img, fsaverage.pial_left)
texture_right = surface.vol_to_surf(new_img, fsaverage.pial_right)

def plot_interactive_surface(coords, faces, texture, title):
    x, y, z = coords.T
    i, j, k = faces.T

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                intensity=texture,
                colorscale="RdBu",
                showscale=True,
                opacity=1.0
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        width=800,
        height=800
    )
    return fig

fig_left = plot_interactive_surface(
    coords_left, faces_left,
    texture_left,
    "Functional Connectivity changes between LSD and Placebo (left hemisphere)"
)

fig_right = plot_interactive_surface(
    coords_right, faces_right,
    texture_right,
    "Functional Connectivity changes between LSD and Placebo (right hemisphere)"
)



# Display side-by-side
st.plotly_chart(fig_left, use_container_width=True)
st.plotly_chart(fig_right, use_container_width=True)


