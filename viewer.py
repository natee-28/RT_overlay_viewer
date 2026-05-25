import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from rt_utils import RTStructBuilder
from skimage.transform import resize
import SimpleITK as sitk
from scipy.ndimage import binary_opening, binary_closing
import shutil
import copy
from skimage.segmentation import find_boundaries


def export_full_ct_series(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    for fname in os.listdir(src_folder):
        path = os.path.join(src_folder, fname)

        try:
            ds = pydicom.dcmread(path)

            if getattr(ds, "Modality", "") != "CT":
                continue

            desc = getattr(ds, "SeriesDescription", "")

            if desc != "FULL":
                continue

            shutil.copy2(path, os.path.join(dst_folder, fname))

        except:
            pass


def load_dicom_series(folder):

    slices = []
    shapes = {}
    series_count = {}

    for fname in os.listdir(folder):

        path = os.path.join(folder, fname)

        try:
            ds = pydicom.dcmread(path)

            modality = getattr(ds, "Modality", "")

            # เอาเฉพาะ CT ก่อน
            if modality != "CT":
                continue
            series_key = (
                getattr(ds, "SeriesNumber", "NA"),
                getattr(ds, "SeriesDescription", "NA")
            )

            series_count[series_key] = series_count.get(series_key, 0) + 1
            desc = getattr(ds, "SeriesDescription", "")
            
            if desc !="EMPTY":
                continue
            """   
            if desc != "FULL":
                continue
            """
            """
            print(
                fname,
                "Series:", getattr(ds, "SeriesNumber", "NA"),
                "Desc:", getattr(ds, "SeriesDescription", "NA"),
                "Instance:", getattr(ds, "InstanceNumber", "NA"),
                "IPP:", getattr(ds, "ImagePositionPatient", "NA")
            )
            """
            if not hasattr(ds, "PixelData"):
                continue

            img = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            img = img * slope + intercept

            shape = img.shape
            shapes[shape] = shapes.get(shape, 0) + 1
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "SliceLocation"):
                z = float(ds.SliceLocation)
            else:
                z = float(getattr(ds, "InstanceNumber", 0))

            #z = float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0)))

            slices.append((z, img, fname, shape, ds))

        except Exception as e:
            print("Skip:", fname, e)

    print("Found image shapes:", shapes)

    if len(slices) == 0:
        raise ValueError("No CT DICOM images found")

    # ใช้ shape ที่เจอบ่อยที่สุด
    main_shape = max(shapes, key=shapes.get)
    slices = [s for s in slices if s[3] == main_shape]

    slices.sort(key=lambda x: x[0])
    print("First z:", slices[0][0])
    print("Last z:", slices[-1][0])

    print("First file:", slices[0][2])
    print("Last file:", slices[-1][2])
    volume = np.stack([s[1] for s in slices], axis=-1)
    
    print("CT series summary:")
    for key, count in series_count.items():
        print(key, "count:", count)
    
    headers = [s[4] for s in slices]
    return volume, headers

def print_dicom_geometry(ct_headers, rd):
    ct0 = ct_headers[0]
    ct_last = ct_headers[-1]

    print("\n--- CT Geometry ---")
    print("CT first IPP:", ct0.ImagePositionPatient)
    print("CT last IPP :", ct_last.ImagePositionPatient)
    print("CT PixelSpacing:", ct0.PixelSpacing)
    print("CT SliceThickness:", getattr(ct0, "SliceThickness", "NA"))
    print("CT ImageOrientationPatient:", ct0.ImageOrientationPatient)

    print("\n--- RTDOSE Geometry ---")
    print("RD ImagePositionPatient:", rd.ImagePositionPatient)
    print("RD PixelSpacing:", rd.PixelSpacing)
    print("RD GridFrameOffsetVector first:", rd.GridFrameOffsetVector[0])
    print("RD GridFrameOffsetVector last :", rd.GridFrameOffsetVector[-1])
    print("RD DoseGridScaling:", rd.DoseGridScaling)
    

def find_rtstruct_file(folder):
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            ds = pydicom.dcmread(path)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                return path
        except:
            pass
    return None

def find_rtdose_file(folder):
    for fname in os.listdir(folder):

        path = os.path.join(folder, fname)

        try:
            ds = pydicom.dcmread(path)

            if getattr(ds, "Modality", "") == "RTDOSE":
                return path

        except:
            pass

    return None

def ct_to_sitk(ct, ct_headers):
    ct0 = ct_headers[0]

    spacing_xy = [float(x) for x in ct0.PixelSpacing]
    z_positions = [float(h.ImagePositionPatient[2]) for h in ct_headers]

    if len(z_positions) > 1:
        spacing_z = abs(z_positions[1] - z_positions[0])
    else:
        spacing_z = float(getattr(ct0, "SliceThickness", 1))

    # numpy ct shape = rows, cols, slices
    # sitk needs = slices, rows, cols
    ct_sitk = sitk.GetImageFromArray(np.transpose(ct, (2, 0, 1)))

    ct_sitk.SetSpacing((spacing_xy[1], spacing_xy[0], spacing_z))
    ct_sitk.SetOrigin(tuple(float(x) for x in ct0.ImagePositionPatient))

    direction = [
        float(x) for x in ct0.ImageOrientationPatient
    ]

    row = np.array(direction[:3])
    col = np.array(direction[3:])
    normal = np.cross(row, col)

    direction_3d = np.column_stack((row, col, normal)).flatten()
    ct_sitk.SetDirection(tuple(direction_3d))

    return ct_sitk

def rtdose_to_sitk(rd):
    
    dose = rd.pixel_array.astype(np.float32) * float(rd.DoseGridScaling)

    print("RD Rows:", rd.Rows)
    print("RD Columns:", rd.Columns)
    print("RD NumberOfFrames:", getattr(rd, "NumberOfFrames", "NA"))
    print("RD pixel_array shape:", dose.shape)

    # pydicom RTDOSE usually gives: frames, rows, cols
    dose_sitk = sitk.GetImageFromArray(dose)
    #dose2 = np.transpose(dose, (0, 2, 1))

    #print("Transposed RD shape:", dose2.shape)

    #dose_sitk = sitk.GetImageFromArray(dose2)
    spacing_xy = [float(x) for x in rd.PixelSpacing]

    offsets = np.array(rd.GridFrameOffsetVector, dtype=np.float32)
    spacing_z = float(abs(offsets[1] - offsets[0]))
    #spacing_z = 3.0
    Z_SHIFT = +60  # ลอง +10, -10, +20, -20 mm

    dose_sitk.SetOrigin((
        float(rd.ImagePositionPatient[0]),
        float(rd.ImagePositionPatient[1]),
        float(rd.ImagePositionPatient[2]) + Z_SHIFT
    ))
    
    dose_sitk.SetSpacing((spacing_xy[1], spacing_xy[0], spacing_z))
    #dose_sitk.SetOrigin(tuple(float(x) for x in rd.ImagePositionPatient))
    # temporary alignment hack
    """
    dose_origin = list(float(x) for x in rd.ImagePositionPatient)

    dose_origin[0] -= 57     # x shift
    dose_origin[1] -= 130    # y shift
    """
    #dose_sitk.SetOrigin(tuple(dose_origin))
    dose_sitk.SetOrigin((
        float(rd.ImagePositionPatient[0]),
        float(rd.ImagePositionPatient[1]),
        float(rd.ImagePositionPatient[2])
    ))
    """
    #dose_sitk.SetDirection((1,0,0, 0,1,0, 0,0,1))
    if hasattr(rd, "ImageOrientationPatient"):
        direction = [float(x) for x in rd.ImageOrientationPatient]
        row = np.array(direction[:3])
        col = np.array(direction[3:])
        normal = np.cross(row, col)

        direction_3d = np.column_stack((row, col, normal)).flatten()

        dose_sitk.SetDirection(tuple(direction_3d))

    else:
        dose_sitk.SetDirection((1,0,0,0,1,0,0,0,1))
    """    
    dose_sitk.SetDirection((1,0,0,0,1,0,0,0,1))    

    return dose_sitk


def resample_dose_to_ct(dose_sitk, ct_sitk):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())

    dose_on_ct_sitk = resampler.Execute(dose_sitk)
    dose_on_ct_raw = sitk.GetArrayFromImage(dose_on_ct_sitk)
    print("SITK raw shape:", dose_on_ct_raw.shape)

    dose_on_ct = np.transpose(dose_on_ct_raw, (1, 2, 0))
    print("SITK transposed shape:", dose_on_ct.shape)

    dose_on_ct = sitk.GetArrayFromImage(dose_on_ct_sitk)

    # sitk array = slices, rows, cols
    # convert back to rows, cols, slices
    dose_on_ct = np.transpose(dose_on_ct, (1, 2, 0))

    return dose_on_ct

def export_overlay_ct_series(
    output_folder,
    ct_headers,
    ct_volume,
    dose,
    rd,
    ct_headers_for_spacing
):

    os.makedirs(output_folder, exist_ok=True)

    global_max = np.max(dose)
    
    new_series_uid = pydicom.uid.generate_uid()  #gen once 

    for i in range(ct_volume.shape[2]):

        ds = copy.deepcopy(ct_headers[i])

        img = ct_volume[:, :, i].copy()

        # ---------------- dose frame mapping ----------------
        dose_i = int(i * dose.shape[0] / ct_volume.shape[2])
        dose_i = min(max(dose_i, 0), dose.shape[0] - 1)

        raw_dose = dose[dose_i, :, :]

        dose_norm = raw_dose / global_max

        mask_mid = dose_norm >= 0.65
        mask_high = dose_norm >= 0.90

        # ---------------- resize ----------------
        ct_ps = [float(x) for x in ct_headers_for_spacing[0].PixelSpacing]
        rd_ps = [float(x) for x in rd.PixelSpacing]

        scale_row = rd_ps[0] / ct_ps[0]
        scale_col = rd_ps[1] / ct_ps[1]

        new_rows = int(raw_dose.shape[0] * scale_row)
        new_cols = int(raw_dose.shape[1] * scale_col)

        mask_mid_rs = resize(
            mask_mid.astype(np.float32),
            output_shape=(new_rows, new_cols),
            preserve_range=True,
            anti_aliasing=False
        ) > 0.5

        mask_high_rs = resize(
            mask_high.astype(np.float32),
            output_shape=(new_rows, new_cols),
            preserve_range=True,
            anti_aliasing=False
        ) > 0.5

        canvas_mid = np.zeros(img.shape, dtype=bool)
        canvas_high = np.zeros(img.shape, dtype=bool)

        r0 = (img.shape[0] - new_rows) // 2
        c0 = (img.shape[1] - new_cols) // 2

        canvas_mid[r0:r0+new_rows, c0:c0+new_cols] = mask_mid_rs
        canvas_high[r0:r0+new_rows, c0:c0+new_cols] = mask_high_rs

        # ---------------- burn thin line ----------------
        line_mid = find_boundaries(canvas_mid, mode="outer")
        line_high = find_boundaries(canvas_high, mode="outer")

        img_out = img.copy()

        img_out[line_mid] = 700
        img_out[line_high] = 1200
        #img_out = img.copy()

        #img_out[canvas_mid] = 700
        #img_out[canvas_high] = 1200

        # ---------------- DICOM update ----------------
        #ds.PixelData = img_out.astype(np.int16).tobytes()
        # Convert HU back to original stored pixel value
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))

        stored = (img_out - intercept) / slope
        stored = np.round(stored).astype(np.int16)

        ds.PixelData = stored.tobytes()

        ds.SeriesDescription = "RT_OVERLAY_TEST"
        
        ds.SeriesNumber = 9002

        ds.SeriesInstanceUID = new_series_uid           #pydicom.uid.generate_uid()

        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        
        ds.InstanceNumber = i + 1

        out_name = os.path.join(
            output_folder,
            f"CT_OVERLAY_{i:04d}.dcm"
        )

        ds.save_as(out_name)

        print("Saved:", out_name)
        
def export_color_overlay_ct_series(
    output_folder,
    ct_headers,
    ct_volume,
    dose,
    rd,
    ct_headers_for_spacing
):

    os.makedirs(output_folder, exist_ok=True)

    global_max = np.max(dose)
    new_series_uid = pydicom.uid.generate_uid()

    for i in range(ct_volume.shape[2]):

        ds = copy.deepcopy(ct_headers[i])
        img = ct_volume[:, :, i].copy()

        # ---------------- CT grayscale to 8-bit ----------------
        WL = 40
        WW = 400

        img8 = np.clip((img - (WL - WW / 2)) / WW * 255, 0, 255).astype(np.uint8)

        rgb = np.stack([img8, img8, img8], axis=-1)

        # ---------------- dose frame mapping ----------------
        dose_i = int(i * dose.shape[0] / ct_volume.shape[2])
        dose_i = min(max(dose_i, 0), dose.shape[0] - 1)

        raw_dose = dose[dose_i, :, :]
        dose_norm = raw_dose / global_max

        mask_mid = dose_norm >= 0.65
        mask_high = dose_norm >= 0.90

        # ---------------- resize ----------------
        ct_ps = [float(x) for x in ct_headers_for_spacing[0].PixelSpacing]
        rd_ps = [float(x) for x in rd.PixelSpacing]

        scale_row = rd_ps[0] / ct_ps[0]
        scale_col = rd_ps[1] / ct_ps[1]

        new_rows = int(raw_dose.shape[0] * scale_row)
        new_cols = int(raw_dose.shape[1] * scale_col)

        mask_mid_rs = resize(
            mask_mid.astype(np.float32),
            output_shape=(new_rows, new_cols),
            preserve_range=True,
            anti_aliasing=False
        ) > 0.5

        mask_high_rs = resize(
            mask_high.astype(np.float32),
            output_shape=(new_rows, new_cols),
            preserve_range=True,
            anti_aliasing=False
        ) > 0.5

        canvas_mid = np.zeros(img.shape, dtype=bool)
        canvas_high = np.zeros(img.shape, dtype=bool)

        r0 = (img.shape[0] - new_rows) // 2
        c0 = (img.shape[1] - new_cols) // 2

        canvas_mid[r0:r0+new_rows, c0:c0+new_cols] = mask_mid_rs
        canvas_high[r0:r0+new_rows, c0:c0+new_cols] = mask_high_rs

        line_mid = find_boundaries(canvas_mid, mode="outer")
        line_high = find_boundaries(canvas_high, mode="outer")

        # yellow line
        rgb[line_mid, 0] = 255
        rgb[line_mid, 1] = 255
        rgb[line_mid, 2] = 0

        # red line
        rgb[line_high, 0] = 255
        rgb[line_high, 1] = 0
        rgb[line_high, 2] = 0

        # ---------------- DICOM RGB update ----------------
        ds.PixelData = rgb.tobytes()

        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0

        if "RescaleSlope" in ds:
            del ds.RescaleSlope
        if "RescaleIntercept" in ds:
            del ds.RescaleIntercept

        ds.SeriesDescription = "RT_COLOR_OVERLAY_TEST"
        ds.SeriesNumber = 9003
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.InstanceNumber = i + 1

        out_name = os.path.join(
            output_folder,
            f"CT_COLOR_OVERLAY_{i:04d}.dcm"
        )

        ds.save_as(out_name)

        print("Saved:", out_name)
# -----------------------------
# MAIN
# -----------------------------
"""
if __name__ == "__main__":


folder = input("Enter DICOM folder path: ")

#ct_folder = r"D:\RT_project\CT_FULL"

#export_full_ct_series(folder, ct_folder)

ct, ct_headers = load_dicom_series(folder)


rs_path = find_rtstruct_file(folder)

if rs_path is None:
    print("No RTSTRUCT file found")
else:
    print("RTSTRUCT found:", rs_path)

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=folder,
        rt_struct_path=rs_path
    )

    roi_names = rtstruct.get_roi_names()
    
    body_mask = rtstruct.get_roi_mask_by_name("BODY")

    print("BODY mask shape:", body_mask.shape)

    print("ROI names:")
    for i, name in enumerate(roi_names):
        print(i, name)

print("Volume shape:", ct.shape)

rd_path = find_rtdose_file(folder)

if rd_path is None:
    print("No RTDOSE found")
    dose_ct = np.zeros_like(ct, dtype=np.float32)

else:
    print("RTDOSE found:", rd_path)

    rd = pydicom.dcmread(rd_path)

    dose = rd.pixel_array.astype(np.float32)
    scaling = float(rd.DoseGridScaling)
    dose = dose * scaling

    print("Dose shape:", dose.shape)
    print("Dose max:", np.max(dose))

    print_dicom_geometry(ct_headers, rd)
    
    ct_sitk = ct_to_sitk(ct, ct_headers)
    dose_sitk = rtdose_to_sitk(rd)

    dose_ct = resample_dose_to_ct(dose_sitk, ct_sitk)
    print("SITK dose_ct frame check:")
    for k in [0, 50, 100, 150, 200, 250, 300, 350, 400, 425]:
        if k < dose_ct.shape[2]:
            print(
                "CT slice:", k,
                "max:", np.max(dose_ct[:, :, k]),
                "sum:", np.sum(dose_ct[:, :, k])
            )

    print("Dose on CT shape:", dose_ct.shape)
    print("Dose on CT max:", np.max(dose_ct))
    print("Dose on CT min:", np.min(dose_ct))
    print("Dose on CT mean:", np.mean(dose_ct)) 
    
    print_dicom_geometry(ct_headers, rd)
    print("RD ImageOrientationPatient:",
      getattr(rd, "ImageOrientationPatient", "NONE"))
    
    print("Raw RD frame check:")
    for k in [0, 25, 50, 75, 100, 125, 150, 175, 200, 212]:
        print(k, np.max(dose[k, :, :]), np.sum(dose[k, :, :]))


#dose_resized = dose_resized.astype(np.float32)

#print("Resized dose shape:", dose_resized.shape)
# show middle slice
# -----------------------------
# Interactive CT viewer
# -----------------------------
state = {
    "WL": 40,
    "WW": 400,
    "slice_idx": ct.shape[2] // 2
}

fig, ax = plt.subplots()

def update_display():

    i = state["slice_idx"]
    WL = state["WL"]
    WW = state["WW"]

    ax.clear()

    # ---------------- CT ----------------
    ax.imshow(
        ct[:, :, i],
        cmap="gray",
        vmin=WL - WW / 2,
        vmax=WL + WW / 2
    )

    # ---------------- SITK DOSE ----------------

    # ---------------- RAW DOSE MASK ROUTE ----------------
    dose_i = int(i * dose.shape[0] / ct.shape[2])
    dose_i = min(max(dose_i, 0), dose.shape[0] - 1)

    raw_dose = dose[dose_i, :, :]

    dose_norm_raw = raw_dose / np.max(dose)

# high / moderate dose mask
    mask_mid = dose_norm_raw >= 0.65
    mask_high = dose_norm_raw >= 0.90

    ct_ps = [float(x) for x in ct_headers[0].PixelSpacing]
    rd_ps = [float(x) for x in rd.PixelSpacing]

    scale_row = rd_ps[0] / ct_ps[0]
    scale_col = rd_ps[1] / ct_ps[1]

    new_rows = int(raw_dose.shape[0] * scale_row)
    new_cols = int(raw_dose.shape[1] * scale_col)

    mask_mid_rs = resize(
    mask_mid.astype(np.float32),
    output_shape=(new_rows, new_cols),
    preserve_range=True,
    anti_aliasing=False
    ) > 0.5

    mask_high_rs = resize(
    mask_high.astype(np.float32),
    output_shape=(new_rows, new_cols),
    preserve_range=True,
    anti_aliasing=False
    ) > 0.5

    canvas_mid = np.zeros(ct[:, :, i].shape, dtype=np.float32)
    canvas_high = np.zeros(ct[:, :, i].shape, dtype=np.float32)

    r0 = (canvas_mid.shape[0] - new_rows) // 2
    c0 = (canvas_mid.shape[1] - new_cols) // 2

    canvas_mid[r0:r0+new_rows, c0:c0+new_cols] = mask_mid_rs
    canvas_high[r0:r0+new_rows, c0:c0+new_cols] = mask_high_rs

    print(
        "CT slice:", i,
        "RD frame:", dose_i,
        "raw max:", np.max(raw_dose),
        "norm max:", np.max(dose_norm_raw)
    )

    if np.any(canvas_mid):
        ax.contour(
            canvas_mid,
            levels=[0.5],
            colors=["y"],
            linewidths=1
        )

    if np.any(canvas_high):
        ax.contour(
            canvas_high,
            levels=[0.5],
            colors=["r"],
            linewidths=1
        )
 
    
    # ---------------- BODY ----------------
    
    ax.contour(
        body_mask[:, :, i],
        levels=[0.5],
        colors="r",
        linewidths=1
    )
    
    ax.set_title(
        f"Slice {i+1}/{ct.shape[2]} | WL={WL}, WW={WW}"
    )

    ax.axis("off")

    fig.canvas.draw_idle()
    
def on_key(event):

    if event.key == "up":
        state["slice_idx"] = min(state["slice_idx"] + 1, ct.shape[2] - 1)

    elif event.key == "down":
        state["slice_idx"] = max(state["slice_idx"] - 1, 0)

    elif event.key == "right":
        state["WL"] += 10

    elif event.key == "left":
        state["WL"] -= 10

    elif event.key in ["+", "="]:
        state["WW"] += 20

    elif event.key in ["-", "_"]:
        state["WW"] = max(20, state["WW"] - 20)

    elif event.key == "1":
        state["WL"], state["WW"] = 40, 400

    elif event.key == "2":
        state["WL"], state["WW"] = 300, 1500

    elif event.key == "3":
        state["WL"], state["WW"] = -600, 1500

    update_display()


fig.canvas.mpl_connect("key_press_event", on_key)

update_display()

output_folder=r"D:\RT_project\EXPORT_CT"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

export_overlay_ct_series(
    output_folder=r"D:\RT_project\EXPORT_CT",
    ct_headers=ct_headers,
    ct_volume=ct,
    dose=dose,
    ct_headers_for_spacing=ct_headers
)

output_folder = r"D:\RT_project\EXPORT_CT_COLOR"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

export_color_overlay_ct_series(
    output_folder=output_folder,
    ct_headers=ct_headers,
    ct_volume=ct,
    dose=dose,
    ct_headers_for_spacing=ct_headers
)


plt.show()
"""