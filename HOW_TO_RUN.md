# HOW TO RUN — Multimodal Human Activity Recognition (UTD-MHAD)

## Prerequisites

- Python 3.9+
- pip

## Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Download the dataset** from the [UTD-MHAD Google Drive](https://drive.google.com/drive/folders/1iMZ68ASWTvBomP2U10qhAIsaQ2z-XTta):

   Download the following files and place them in `data/`:
   - `Inertial.zip`
   - `RGB-part1.zip`, `RGB-part2.zip`, `RGB-part3.zip`, `RGB-part4.zip`

   Alternatively, use `gdown`:
   ```bash
   pip install gdown
   cd data
   gdown --folder "https://drive.google.com/drive/folders/1iMZ68ASWTvBomP2U10qhAIsaQ2z-XTta"
   ```

3. **Extract the data:**

```bash
cd data

# Extract inertial data
unzip "UTD-MHAD Dataset/Inertial.zip" -d .

# Extract RGB videos
for f in "UTD-MHAD Dataset"/RGB-part*.zip; do unzip -o "$f" -d .; done

# Organize RGB files into subdirectory
mkdir -p RGB
mv *.avi RGB/
```

After extraction, the directory structure should be:
```
data/
  RGB/           # 861 .avi files (e.g., a1_s1_t1_color.avi)
  Inertial/      # 861 .mat files (e.g., a1_s1_t1_inertial.mat)
```

## Running the Notebooks

### Part 1: Exploratory Data Analysis

```bash
cd notebooks
jupyter notebook part1_eda.ipynb
```

Or run non-interactively:
```bash
jupyter nbconvert --to notebook --execute notebooks/part1_eda.ipynb --output part1_eda.ipynb
```

## Project Structure

```
multimodal_har/
  data/
    RGB/              # RGB video files (.avi)
    Inertial/         # Inertial sensor files (.mat)
    subset_metadata.csv  # Generated metadata for selected subset
  notebooks/
    part1_eda.ipynb   # Part 1: Data loading & EDA
  requirements.txt    # Python dependencies
  HOW_TO_RUN.md       # This file
```
