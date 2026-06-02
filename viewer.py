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
from pydicom.dataset import FileMetaDataset
from pydicom.sequence import Sequence
import datetime
import highdicom as hd


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
    ct_headers_for_spacing,
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
    saved_ct_overlay_headers = []
    
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
        saved_ct_overlay_headers.append(copy.deepcopy(ds))
        
        #ps_name = os.path.join(output_folder, f"GSPS_COLOR_OVERLAY_{i:04d}.dcm")
        #create_gsps_for_overlay_ct(ds, ps_name)
    #gsps_path = os.path.join(output_folder, "GSPS_COLOR_OVERLAY_ALL.dcm")
    #create_gsps_for_overlay_seriesG(saved_ct_overlay_headers, gsps_path)  #****************************************************
def export_dicom_segmentation(
    output_folder,
    ct_headers,
    dose,
    rd,
    ct_headers_for_spacing
):

    os.makedirs(output_folder, exist_ok=True)

    global_max = np.max(dose)

    masks = []

    for i in range(len(ct_headers)):

        ds = ct_headers[i]

        dose_i = int(i * dose.shape[0] / len(ct_headers))
        dose_i = min(max(dose_i, 0), dose.shape[0] - 1)

        raw_dose = dose[dose_i, :, :]
        dose_norm = raw_dose / global_max

        mask70 = dose_norm >= 0.70
        mask95 = dose_norm >= 0.90

        ct0 = ct_headers_for_spacing[0]

        ct_ps = [float(x) for x in ct0.PixelSpacing]
        rd_ps = [float(x) for x in rd.PixelSpacing]

        scale_row = rd_ps[0] / ct_ps[0]
        scale_col = rd_ps[1] / ct_ps[1]

        new_rows = int(raw_dose.shape[0] * scale_row)
        new_cols = int(raw_dose.shape[1] * scale_col)

        mask70_rs = resize(
            mask70.astype(np.float32),
            output_shape=(new_rows, new_cols),
            preserve_range=True,
            anti_aliasing=False
        ) > 0.5

        mask95_rs = resize(
            mask95.astype(np.float32),
            output_shape=(new_rows, new_cols),
            preserve_range=True,
            anti_aliasing=False
        ) > 0.5

        ct_origin = np.array(ct0.ImagePositionPatient, dtype=float)
        rd_origin = np.array(rd.ImagePositionPatient, dtype=float)

        delta = rd_origin - ct_origin

        r0 = int(round(delta[1] / ct_ps[0]))
        c0 = int(round(delta[0] / ct_ps[1]))

        canvas70 = np.zeros((ds.Rows, ds.Columns), dtype=np.uint8)
        canvas95 = np.zeros((ds.Rows, ds.Columns), dtype=np.uint8)

        r1 = max(r0, 0)
        c1 = max(c0, 0)

        r2 = min(r0 + new_rows, ds.Rows)
        c2 = min(c0 + new_cols, ds.Columns)

        mr1 = r1 - r0
        mc1 = c1 - c0

        mr2 = mr1 + (r2 - r1)
        mc2 = mc1 + (c2 - c1)

        if r2 > r1 and c2 > c1:

            canvas70[r1:r2, c1:c2] = mask70_rs[mr1:mr2, mc1:mc2]
            canvas95[r1:r2, c1:c2] = mask95_rs[mr1:mr2, mc1:mc2]

        masks.append(np.stack([canvas70, canvas95], axis=-1))

    pixel_array = np.stack(masks, axis=0)

    print("SEG pixel_array shape:", pixel_array.shape)
# ====================================================================
# Segment Sescription 
#====================================================================
    segment70 = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label="Dose70",
        segmented_property_category=hd.sr.CodedConcept(
            value="T-D0050",
            scheme_designator="SRT",
            meaning="Tissue"
        ),
        segmented_property_type=hd.sr.CodedConcept(
            value="T-D0050",
            scheme_designator="SRT",
            meaning="Dose Region"
        ),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        #algorithm_name="RT_Dose_System"
    )

    segment95 = hd.seg.SegmentDescription(
        segment_number=2,
        segment_label="Dose90",
        segmented_property_category=hd.sr.CodedConcept(
            value="T-D0050",
            scheme_designator="SRT",
            meaning="Tissue"
        ),
        segmented_property_type=hd.sr.CodedConcept(
            value="T-D0050",
            scheme_designator="SRT",
            meaning="Dose Region"
        ),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        #algorithm_name="RT_Dose_System"
    )
# =====================================================================
# Creat SEG DataSET
# ======================================================================
    seg = hd.seg.Segmentation(
        source_images=ct_headers,
        pixel_array=pixel_array,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[
            segment70,
            segment95
        ],
        series_instance_uid=pydicom.uid.generate_uid(),
        series_number=9007,
        sop_instance_uid=pydicom.uid.generate_uid(),
        instance_number=1,
        manufacturer="RT_PROJECT",
        manufacturer_model_name="DoseSeg",
        software_versions="1.0",
        device_serial_number="001"
    )

    out_file = os.path.join(
        output_folder,
        "AI_DOSE_SEGMENTATION.dcm"
    )

    seg.save_as(out_file)

    print("Saved:", out_file)
    
"""
GSPS_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.11.1"

def create_gsps_for_overlay_ct(ct_ds, output_path):
    
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = GSPS_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    gsps = FileDataset(
        output_path,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )

    gsps.is_little_endian = True
    gsps.is_implicit_VR = False

    # Patient / Study
    for tag in [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "StudyInstanceUID", "StudyDate", "StudyTime",
        "StudyID", "AccessionNumber", "ReferringPhysicianName"
    ]:
        if hasattr(ct_ds, tag):
            setattr(gsps, tag, getattr(ct_ds, tag))

    # PR Series = 9006
    gsps.Modality = "PR"
    gsps.SeriesInstanceUID = pydicom.uid.generate_uid()
    gsps.SeriesNumber = 9006
    gsps.SeriesDescription = "GSPS_COLOR_FOR_OVERLAY_9005"

    gsps.SOPClassUID = GSPS_SOP_CLASS
    gsps.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    gsps.InstanceNumber = 1

    now = datetime.datetime.now()
    gsps.ContentLabel = "DOSE_COLOR"
    gsps.ContentDescription = "Color layer mapping for DICOM overlays"
    gsps.ContentCreatorName = "RT_OVERLAY_VIEWER"
    gsps.PresentationCreationDate = now.strftime("%Y%m%d")
    gsps.PresentationCreationTime = now.strftime("%H%M%S")

    # Reference CT+Overlay 9005 image
    ref_img = Dataset()
    ref_img.ReferencedSOPClassUID = ct_ds.SOPClassUID
    ref_img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = ct_ds.SeriesInstanceUID
    ref_series.ReferencedImageSequence = Sequence([ref_img])

    gsps.ReferencedSeriesSequence = Sequence([ref_series])

    # Displayed Area
    area = Dataset()
    area.ReferencedImageSequence = Sequence([ref_img])
    area.DisplayedAreaTopLeftHandCorner = [1, 1]
    area.DisplayedAreaBottomRightHandCorner = [int(ct_ds.Columns), int(ct_ds.Rows)]
    area.PresentationSizeMode = "SCALE TO FIT"
    area.PresentationPixelAspectRatio = [1, 1]
    gsps.DisplayedAreaSelectionSequence = Sequence([area])

    # Presentation LUT
    lut = Dataset()
    lut.PresentationLUTShape = "IDENTITY"
    gsps.SoftcopyPresentationLUTSequence = Sequence([lut])

    # Graphic Layers with color
    layer70 = Dataset()
    layer70.GraphicLayer = "LAYER_DOSE_70"
    layer70.GraphicLayerOrder = 1
    layer70.GraphicLayerRecommendedDisplayCIELabValue = [65535, 0, 32768]  # green-ish
    layer70.GraphicLayerDescription = "Dose boundary 70"

    layer90 = Dataset()
    layer90.GraphicLayer = "LAYER_DOSE_90"
    layer90.GraphicLayerOrder = 2
    layer90.GraphicLayerRecommendedDisplayCIELabValue = [65535, 32768, 32768]  # red-ish
    layer90.GraphicLayerDescription = "Dose boundary 90"

    gsps.GraphicLayerSequence = Sequence([layer70, layer90])

    # Overlay Activation: map 6000/6002 to layers
    gsps.add_new((0x6000, 0x1001), "CS", "LAYER_DOSE_70")
    gsps.add_new((0x6002, 0x1001), "CS", "LAYER_DOSE_90")

    gsps.save_as(output_path, write_like_original=False)

def create_gsps_for_overlay_series(ct_ds_list, output_path):
    
    if not ct_ds_list:
        print("Error: ct_ds_list is empty")
        return
    
    first_ct = ct_ds_list[0]
    GSPS_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.11.1"
    
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 222
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    
    file_meta.MediaStorageSOPClassUID = GSPS_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.2.1143.100.1' # pydicom.uid.generate_uid()

    gsps = FileDataset(
        output_path,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )

    gsps.is_little_endian = True
    gsps.is_implicit_VR = False

    # Patient / Study
    for tag in [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "StudyInstanceUID", "StudyDate", "StudyTime",
        "StudyID", "AccessionNumber", "ReferringPhysicianName"
    ]:
        if hasattr(first_ct, tag):
            setattr(gsps, tag, getattr(first_ct, tag))

    # PR Series = 9006
    gsps.Modality = "PR"
    gsps.SeriesInstanceUID = pydicom.uid.generate_uid()
    gsps.SeriesNumber = 9006
    gsps.SeriesDescription = "GSPS_COLOR_MAPPING_9005"

    gsps.SOPClassUID = GSPS_SOP_CLASS
    gsps.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    gsps.InstanceNumber = 1

    now = datetime.datetime.now()
    gsps.ContentLabel = "DOSE_COLOR"
    gsps.ContentDescription = "Color layer mapping for DICOM overlays"
    gsps.ContentCreatorName = "RT_OVERLAY_VIEWER"
    gsps.PresentationCreationDate = now.strftime("%Y%m%d")
    gsps.PresentationCreationTime = now.strftime("%H%M%S")

    # Reference all CT+Overlay slices
    ref_images = []
    for ct_ds in ct_ds_list:
        ref_img = Dataset()
        ref_img.ReferencedSOPClassUID = ct_ds.SOPClassUID
        ref_img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        ref_images.append(ref_img)

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = first_ct.SeriesInstanceUID
    ref_series.ReferencedImageSequence = Sequence(ref_images)
    gsps.ReferencedSeriesSequence = Sequence([ref_series])

    # Displayed Area for all referenced images
    area = Dataset()
    area.ReferencedImageSequence = Sequence(ref_images)
    area.DisplayedAreaTopLeftHandCorner = [1, 1]
    area.DisplayedAreaBottomRightHandCorner = [
        int(first_ct.Columns),
        int(first_ct.Rows)
    ]
    area.PresentationSizeMode = "SCALE TO FIT"
    area.PresentationPixelAspectRatio = [1, 1]
    #area.ReferencedImageSequence = Sequence(ref_images) 
    if hasattr(first_ct, 'PixelSpacing'):
        area.PresentationPixelSpacing = first_ct.PixelSpacing
        
    gsps.DisplayedAreaSelectionSequence = Sequence([area])

    # Presentation LUT
    gsps.PresentationLUTShape = "IDENTITY"

    # Graphic layers
    layer70 = Dataset()
    layer70.GraphicLayer = "LAYER_DOSE_70"
    layer70.GraphicLayerOrder = 1
    layer70.GraphicLayerDescription = "Dose boundary 70"
    layer70.GraphicLayerRecommendedDisplayCIELABValue = [65535, 0, 32768]

    layer90 = Dataset()
    layer90.GraphicLayer = "LAYER_DOSE_90"
    layer90.GraphicLayerOrder = 2
    layer90.GraphicLayerDescription = "Dose boundary 90"
    layer90.GraphicLayerRecommendedDisplayCIELABValue = [65535, 32768, 32768]

    gsps.GraphicLayerSequence = Sequence([layer70, layer90])

    # Overlay Activation
    gsps.add_new((0x6000, 0x1001), "CS", "LAYER_DOSE_70")
    gsps.add_new((0x6002, 0x1001), "CS", "LAYER_DOSE_90")

    gsps.save_as(output_path, write_like_original=False)
    print("Saved GSPS series:", output_path)

import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

def create_gsps_for_overlay_seriesG(ct_ds_list, output_path):
    if not ct_ds_list:
        print("Error: ct_ds_list is empty")
        return

    first_ct = ct_ds_list[0]
    GSPS_SOP_CLASS = '1.2.840.10008.5.1.4.1.1.11.1' # มาตรฐานคลาส GSPS Storage

    # 1. นิยามข้อมูลหัวไฟล์ (File Meta) ให้เสถียรและระบุสิทธิ์ซอฟต์แวร์ให้ถูกต้อง
    file_meta = Dataset()
    file_meta.FileMetaInformationGroupLength = 222
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = GSPS_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    # ใช้เลขคลาสมาตรฐานสากล ห้ามสุ่มเด็ดขาด ป้องกันระบบคัดกรองขยะ
    file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.2.1143.100.1' 
    file_meta.ImplementationVersionName = 'PYDICOM_GSPS_100'

    # 2. เริ่มต้นโครงสร้างข้อมูล Dataset หลัก
    gsps = FileDataset(output_path, Dataset(), file_meta=file_meta, preamble=b"\0" * 128)
    gsps.is_little_endian = True
    gsps.is_implicit_VR = False

    # คัดลอกแท็กตัวตนและการตรวจ (Patient/Study Tags) จากภาพแรก
    tags_to_copy = [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "StudyInstanceUID", "StudyDate", "StudyTime",
        "StudyID", "AccessionNumber", "ReferringPhysicianName"
    ]
    for tag in tags_to_copy:
        if hasattr(first_ct, tag):
            setattr(gsps, tag, getattr(first_ct, tag))

    # กำหนดลักษณะเฉพาะของซีรีส์ PR ใหม่
    gsps.Modality = "PR"
    gsps.SeriesInstanceUID = pydicom.uid.generate_uid()
    gsps.SeriesNumber = 9006
    gsps.SeriesDescription = "GSPS_COLOR_FOR_OVERLAY_9005"

    gsps.SOPClassUID = GSPS_SOP_CLASS
    gsps.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    gsps.InstanceNumber = 1

    # ตั้งเวลาสร้างเอกสารข้อมูล
    now = datetime.datetime.now()
    gsps.ContentLabel = "DOSE_COLOR"
    gsps.ContentDescription = "Color layer mapping for DICOM overlays"
    gsps.ContentCreatorName = "RT_OVERLAY_VIEWER"
    gsps.PresentationCreationDate = now.strftime("%Y%m%d")
    gsps.PresentationCreationTime = now.strftime("%H%M%S")

    # 3. จัดทำสารบัญภาพอ้างอิง (Referenced Series/Image Sequence) ครบทุกสไลซ์
    ref_images = []
    for ct_ds in ct_ds_list:
        ref_img = Dataset()
        ref_img.ReferencedSOPClassUID = ct_ds.SOPClassUID
        ref_img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        ref_images.append(ref_img)

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = first_ct.SeriesInstanceUID
    ref_series.ReferencedImageSequence = Sequence(ref_images)
    gsps.ReferencedSeriesSequence = Sequence([ref_series])

    # 4. โมดูลบังคับที่ 1: จัดระบบพื้นที่แสดงผล (Displayed Area Module) ให้สมบูรณ์แบบ
    area = Dataset()
    area.ReferencedImageSequence = Sequence(ref_images) # ต้องผูกภาพอ้างอิงในหน้านี้ด้วย
    # พิกัดขอบหน้าจอแบบ Signed Long (SL) ต้องส่งเป็นลิสต์ของตัวเลขจำนวนเต็มจำนวน 2 คู่
    area.DisplayedAreaTopLeftHandCorner = [1, 1]
    area.DisplayedAreaBottomRightHandCorner = [int(first_ct.Columns), int(first_ct.Rows)]
    area.PresentationSizeMode = "SCALE TO FIT"
    area.PresentationPixelAspectRatio = [1, 1]
    
    if hasattr(first_ct, 'PixelSpacing'):
        area.PresentationPixelSpacing = first_ct.PixelSpacing

    gsps.DisplayedAreaSelectionSequence = Sequence([area])

    # 5. โมดูลบังคับที่ 2: นิยามระบบการคุมแสงภาพ (Softcopy VOI LUT Module) ห้ามขาดเด็ดขาด!
    # จุดนี้คือจุดที่ทำให้ซอฟต์แวร์ Philips ล่มบ่อยที่สุดหากไม่มีการประกาศโครงสร้าง
    voi_lut_item = Dataset()
    voi_lut_item.ReferencedImageSequence = Sequence(ref_images)
    # ดึงค่าควบคุมระดับความสว่างหน้าต่างภาพ (Window Center/Width) จากภาพ CT แผ่นแรกมาใช้เป็นมาตรฐาน
    voi_lut_item.WindowCenter = getattr(first_ct, 'WindowCenter', 40)
    voi_lut_item.WindowWidth = getattr(first_ct, 'WindowWidth', 400)
    gsps.SoftcopyVOILUTSequence = Sequence([voi_lut_item])

    # โครงสร้างบังคับสำหรับแปลงสเกลสีจอภาพดั้งเดิม
    gsps.PresentationLUTShape = "IDENTITY"

    # 6. นิยามเลเยอร์สี (แก้ไขชื่อแท็กพิกัดสากลเป็นตัวพิมพ์ใหญ่ CIELABValue เรียบร้อย)
    layer70 = Dataset()
    layer70.GraphicLayer = "LAYER_DOSE_70"
    layer70.GraphicLayerOrder = 1
    layer70.GraphicLayerDescription = "Dose boundary 70"
    layer70.GraphicLayerRecommendedDisplayCIELABValue = [65535, 0, 32768] # สีเขียวในพิกัด CIELAB

    layer90 = Dataset()
    layer90.GraphicLayer = "LAYER_DOSE_90"
    layer90.GraphicLayerOrder = 2
    layer90.GraphicLayerDescription = "Dose boundary 90"
    layer90.GraphicLayerRecommendedDisplayCIELABValue = [65535, 32768, 32768] # สีแดงในพิกัด CIELAB

    gsps.GraphicLayerSequence = Sequence([layer70, layer90])

    # 7. ผูกสัญญาณเปิดโครงสร้างสีผ่าน Overlay Activation Module
    gsps.add_new((0x6000, 0x1001), "CS", "LAYER_DOSE_70")
    gsps.add_new((0x6002, 0x1001), "CS", "LAYER_DOSE_90")

    # บันทึกไฟล์โดยปิดคำสั่งหยวนข้อมูล บังคับให้คอมพายตามโครงสร้างมาตรฐานสากล
    gsps.save_as(output_path, write_like_original=False)
    print("Saved Standard-compliant GSPS series:", output_path)
"""
# -----------------------------
# MAIN
# -----------------------------
