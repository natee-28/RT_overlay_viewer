import os
import shutil
import numpy as np
import pydicom
import streamlit as st

from viewer import (
    load_dicom_series,
    find_rtstruct_file,
    find_rtdose_file,
    #export_overlay_ct_series,
    export_color_overlay_ct_series,
    export_ct_with_overlay_planes,
    #export_dicom_segmentation,
)

st.title("RT Dose Overlay DICOM Exporter")

folder = st.text_input("DICOM folder")

export_type = st.radio(
    "Export type",
    [
        
        "Color DICOM - RGB preview",
        "DICOM Overlay Plane - experimental",
        
    ]
)

if st.button("Run"):

    ct, ct_headers = load_dicom_series(folder)
    st.write("CT shape:", ct.shape)

    rs_path = find_rtstruct_file(folder)
    st.write("RTSTRUCT:", rs_path)

    rd_path = find_rtdose_file(folder)
    st.write("RTDOSE:", rd_path)

    rd = pydicom.dcmread(rd_path)
    dose = rd.pixel_array.astype(np.float32) * float(rd.DoseGridScaling)

    st.write("Dose shape:", dose.shape)
    st.write("Dose max:", float(np.max(dose)))
   
    if export_type == "Color DICOM - RGB preview":
        
        output_folder = r"D:\RT_project\EXPORT_CT_COLOR"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        export_color_overlay_ct_series(
            output_folder=output_folder,
            ct_headers=ct_headers,
            ct_volume=ct,
            dose=dose,
            rd=rd,
            ct_headers_for_spacing=ct_headers
        )
    elif export_type == "DICOM Overlay Plane - experimental":

        output_folder = r"D:\RT_project\EXPORT_CT_OVERLAY_PLANE"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

        export_ct_with_overlay_planes(
            output_folder=output_folder,
            ct_headers=ct_headers,
            ct_volume=ct,
            dose=dose,
            rd=rd,
            ct_headers_for_spacing=ct_headers
        )

    st.success(f"Export completed: {output_folder}")
