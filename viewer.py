import os
import numpy as np
import pydicom
from pydicom.tag import Tag
import matplotlib.pyplot as plt
from rt_utils import RTStructBuilder
from skimage.transform import resize
import SimpleITK as sitk
from scipy.ndimage import binary_opening, binary_closing
import shutil
import copy
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
import datetime


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
    
   
    #dose_sitk.SetOrigin(tuple(float(x) for x in rd.ImagePositionPatient))
    # temporary alignment hack
    
    #dose_sitk.SetOrigin(tuple(dose_origin))
    dose_sitk.SetOrigin((
        float(rd.ImagePositionPatient[0]),
        float(rd.ImagePositionPatient[1]),
        float(rd.ImagePositionPatient[2])
    ))
    
    dose_sitk.SetSpacing((spacing_xy[1], spacing_xy[0], spacing_z))   
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

        mask_mid = dose_norm >= 0.70
        mask_high = dose_norm >= 0.95

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

        #r0 = (img.shape[0] - new_rows) // 2
        #c0 = (img.shape[1] - new_cols) // 2
        ct0 = ct_headers_for_spacing[0]

        ct_origin = np.array(ct0.ImagePositionPatient, dtype=float)
        rd_origin = np.array(rd.ImagePositionPatient, dtype=float)

        ct_ps = [float(x) for x in ct0.PixelSpacing]
        rd_ps = [float(x) for x in rd.PixelSpacing]

# patient coordinate difference in mm
        delta = rd_origin - ct_origin

# for axial CT: row = patient Y, col = patient X
        r0 = int(round(delta[1] / ct_ps[0]))
        c0 = int(round(delta[0] / ct_ps[1]))

        print("Physical paste r0/c0:", r0, c0)

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
        WW = 1024

        img8 = np.clip((img - (WL - WW / 2)) / WW * 255, 0, 255).astype(np.uint8)

        rgb = np.stack([img8, img8, img8], axis=-1)

        # ---------------- dose frame mapping ----------------
        dose_i = int(i * dose.shape[0] / ct_volume.shape[2])
        dose_i = min(max(dose_i, 0), dose.shape[0] - 1)

        raw_dose = dose[dose_i, :, :]
        dose_norm = raw_dose / global_max
        

        mask_mid = dose_norm >= 0.70
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

        #r0 = (img.shape[0] - new_rows) // 2
        #c0 = (img.shape[1] - new_cols) // 2
        
        ct0 = ct_headers_for_spacing[0]

        ct_origin = np.array(ct0.ImagePositionPatient, dtype=float)
        rd_origin = np.array(rd.ImagePositionPatient, dtype=float)

        ct_ps = [float(x) for x in ct0.PixelSpacing]
        rd_ps = [float(x) for x in rd.PixelSpacing]

# patient coordinate difference in mm
        delta = rd_origin - ct_origin

# for axial CT: row = patient Y, col = patient X
        r0 = int(round(delta[1] / ct_ps[0]))
        c0 = int(round(delta[0] / ct_ps[1]))

        print("Physical paste r0/c0:", r0, c0)

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
# add viewer.py 

def pack_overlay(mask):
    bits = np.asarray(mask, dtype=np.uint8).ravel(order="C")
    packed = np.packbits(bits, bitorder="little")
    if len(packed) % 2 != 0:
        packed = np.append(packed, 0).astype(np.uint8)
    return packed.tobytes()

def add_overlay_plane(ds, group, mask, label):
    ds.add_new(Tag(group, 0x0010), "US", int(ds.Rows))
    ds.add_new(Tag(group, 0x0011), "US", int(ds.Columns))
    #ds.add_new(Tag(group, 0x0022), "LO", label)
    ds.add_new(Tag(group, 0x0040), "CS", "G")
    #ds.add_new(Tag(group, 0x0045), "LO", "AUTOMATED")  # old fashion  
    ds.add_new(Tag(group, 0x0050), "SS", [1, 1])
    ds.add_new(Tag(group, 0x0100), "US", 1)
    ds.add_new(Tag(group, 0x0102), "US", 0)
    ds.add_new(Tag(group, 0x1500), "LO", label)
    ds.add_new(Tag(group, 0x3000), "OB", pack_overlay(mask))

def export_ct_with_overlay_planes(
    output_folder,
    ct_headers,
    ct_volume,
    dose,
    rd,
    ct_headers_for_spacing
):
    os.makedirs(output_folder, exist_ok=True)
    new_series_uid = pydicom.uid.generate_uid()
    global_max = np.max(dose)

    for i in range(ct_volume.shape[2]):
        ds = copy.deepcopy(ct_headers[i])

        dose_i = int(i * dose.shape[0] / ct_volume.shape[2])
        dose_i = min(max(dose_i, 0), dose.shape[0] - 1)

        raw_dose = dose[dose_i, :, :]
        dose_norm = raw_dose / global_max

        mask_mid = dose_norm >= 0.70
        mask_high = dose_norm >= 0.90

        ct0 = ct_headers_for_spacing[0]
        ct_ps = [float(x) for x in ct0.PixelSpacing]
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

        ct_origin = np.array(ct0.ImagePositionPatient, dtype=float)
        rd_origin = np.array(rd.ImagePositionPatient, dtype=float)
        delta = rd_origin - ct_origin

        r0 = int(round(delta[1] / ct_ps[0]))
        c0 = int(round(delta[0] / ct_ps[1]))

        canvas_mid = np.zeros((ds.Rows, ds.Columns), dtype=bool)
        canvas_high = np.zeros((ds.Rows, ds.Columns), dtype=bool)

        r1 = max(r0, 0)
        c1 = max(c0, 0)
        r2 = min(r0 + new_rows, ds.Rows)
        c2 = min(c0 + new_cols, ds.Columns)

        mr1 = r1 - r0
        mc1 = c1 - c0
        mr2 = mr1 + (r2 - r1)
        mc2 = mc1 + (c2 - c1)

        if r2 > r1 and c2 > c1:
            canvas_mid[r1:r2, c1:c2] = mask_mid_rs[mr1:mr2, mc1:mc2]
            canvas_high[r1:r2, c1:c2] = mask_high_rs[mr1:mr2, mc1:mc2]

        line_mid = find_boundaries(canvas_mid, mode="outer")
        line_high = find_boundaries(canvas_high, mode="outer")

        add_overlay_plane(ds, 0x6000, line_mid, "Dose Boundary 70")
        add_overlay_plane(ds, 0x6002, line_high, "Dose Boundary 90")

        ds.SeriesDescription = "CT_WITH_DICOM_OVERLAY_PLANES"
        ds.SeriesNumber = 9005
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.InstanceNumber = i + 1

        out_name = os.path.join(output_folder, f"CT_OVERLAY_PLANE_{i:04d}.dcm")
        ds.save_as(out_name)

# -----------------------------
# MAIN
# -----------------------------
