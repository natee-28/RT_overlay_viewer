# RT Dose Overlay DICOM Exporter

CT + RTDOSE overlay viewer and DICOM exporter.

## Features

- CT + RTDOSE visualization
- Derived DICOM export
- RGB color overlay export
- DICOM Overlay Plane export (6000/6002)
- Preserves CT window/level using Overlay Plane
- Tested with Philips Workstation
- Tested with Fujifilm Synapse PACS

## Export Modes

1. CT grayscale export
2. RGB color overlay export
3. DICOM Overlay Plane export (experimental)

## Tested result:
- DICOM Overlay Plane was successfully displayed on Philips Workstation and Synapse PACS.
- RGB export is provided for fixed color visualization.
- GSPS and DICOM SEG were explored but not used in the final workflow due to limited workstation/PACS rendering support.

---

### Installation

1. Install Miniconda  
https://www.anaconda.com/docs/getting-started/miniconda/main

2. Open Anaconda Prompt

3. Navigate to project folder

4. Create environment and install packages

```bash
conda create -n rt_env python=3.11
conda activate rt_env

pip install -r requirements.txt
```
#### Run 
```bash
run_app.bat
```
## Example Output

![Quick DICOM Preview](Screenshot1.png)
![Quick DICOM Preview](Screenshot2.png)

## Notes

Window/Level functionality was preserved because original CT pixel data remained unchanged.
