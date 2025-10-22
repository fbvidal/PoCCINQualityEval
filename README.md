# PoCCINQualityEval

Proof-of-Concept application for analyzing Brazilian National ID Card images. This tool extracts color palettes, detects faces using multiple DeepFace detectors, identifies facial landmarks, and evaluates compliance with ICAO 9303 standards.

## Features

- **Color Palette Extraction**: Extracts dominant colors with hex codes and percentages using K-Means clustering
- **Multi-Detector Face Recognition**: Uses DeepFace with multiple backends (OpenCV, RetinaFace, MTCNN, SSD)
- **Facial Landmarks Detection**: Identifies eyes, nose, mouth, and other facial features
- **ICAO 9303 Compliance**: Evaluates ID card photos against international standards including:
  - Image resolution requirements
  - Face size and positioning
  - Brightness and contrast levels
  - Face centering and proportions
- **Detailed Reports**: Generates formatted tables showing all evaluation criteria and results

## Installation

### Prerequisites

- Python 3.8 or higher
- uv (fast Python package manager). If you use the setup script below, uv will be installed automatically if missing.

### Quick setup (recommended)

Run the setup script, which creates a virtual environment with uv and installs all dependencies:
```bash
./setup.sh
```

### Manual setup with uv

1. Clone this repository:
```bash
git clone https://github.com/fbvidal/PoCCINQualityEval.git
cd PoCCINQualityEval
```

### Install uv manually (optional)

If you prefer to install uv yourself (the setup script installs it automatically when missing):

- macOS via Homebrew:
```bash
brew install uv
```

- Official installer (macOS/Linux):
```bash
curl -fsSL https://astral.sh/uv/install.sh | sh
# Ensure uv is in PATH for the current shell session
export PATH="$HOME/.local/bin:$PATH"
```

2. Create and activate a virtual environment with uv:
```bash
uv venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate    # Windows
```

3. Install dependencies with uv:
```bash
uv pip install -r requirements.txt
```

**Note**: The first run will download deep learning models for face detection (approximately 100–200MB). This is a one-time download.

## Usage

### Basic Usage

Analyze a single ID card image:
```bash
python id_card_analyzer.py path/to/id_card.jpg
```

### Multiple Images

Analyze multiple ID card images:
```bash
python id_card_analyzer.py card1.jpg card2.png card3.jpg
```

### Output

The tool generates:
1. **Console output**: Real-time analysis progress and results
2. **Text reports**: Saved as `<image_name>_analysis.txt` for each processed image

## Example Output

```
================================================================================
BRAZILIAN NATIONAL ID CARD ANALYSIS REPORT
================================================================================

Image: sample_id_card.jpg

--------------------------------------------------------------------------------
COLOR PALETTE
--------------------------------------------------------------------------------
 Rank Hex Code       RGB Percentage
    1  #1a3c5e  (26, 60, 94)      35.42%
    2  #f0e6d2 (240, 230, 210)    28.15%
    3  #8a7b6c (138, 123, 108)    18.23%
    4  #d4c5a8 (212, 197, 168)    12.67%
    5  #4a5f7c  (74, 95, 124)      5.53%

--------------------------------------------------------------------------------
FACE DETECTION RESULTS
--------------------------------------------------------------------------------
 Detector Detected Confidence
   OPENCV        ✓        0.987
RETINAFACE        ✓        0.945
     MTCNN        ✓        0.923
       SSD        ✓        0.901

--------------------------------------------------------------------------------
FACIAL LANDMARKS
--------------------------------------------------------------------------------
 Feature Count Detected
    Face     1        ✓
    Eyes     2        ✓
   Mouth     1        ✓
    Nose     0        ✗

--------------------------------------------------------------------------------
ICAO 9303 COMPLIANCE EVALUATION
--------------------------------------------------------------------------------
          Criterion     Value      Standard    Result
         Resolution  800x600  Min 600x400  ✓ PASS
         Brightness   145.32       50-240  ✓ PASS
           Contrast    45.67        Max 70  ✓ PASS
  Face Width Ratio     0.625     0.5-0.75  ✓ PASS
 Face Height Ratio     0.725      0.6-0.9  ✓ PASS
  Face Position X      0.498    0.35-0.65  ✓ PASS
  Face Position Y      0.425     0.3-0.6  ✓ PASS

Compliance Summary: 7/7 checks passed
================================================================================
```

## ICAO 9303 Standards

The application evaluates the following ICAO 9303 requirements:

| Criterion | Standard | Description |
|-----------|----------|-------------|
| **Resolution** | Min 600x400 pixels | Minimum image dimensions |
| **Brightness** | 50-240 (0-255 scale) | Average brightness level |
| **Contrast** | Max std dev 70 | Image contrast consistency |
| **Face Width** | 50-75% of image width | Horizontal face proportion |
| **Face Height** | 60-90% of image height | Vertical face proportion |
| **Face Position X** | 35-65% (centered) | Horizontal face centering |
| **Face Position Y** | 30-60% (upper-mid) | Vertical face positioning |

## Project Structure

```
PoCCINQualityEval/
├── id_card_analyzer.py      # Main application
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE                 # License information
```

## Technical Details

### Color Palette Extraction
- Uses K-Means clustering (scikit-learn) to identify dominant colors
- Converts RGB to hex codes
- Calculates color distribution percentages

### Face Detection
The tool uses DeepFace library with multiple detector backends:
- **OpenCV**: Fast, built-in Haar Cascade detector
- **RetinaFace**: High accuracy, single-stage detector
- **MTCNN**: Multi-task Cascaded CNN, good for varied conditions
- **SSD**: Single Shot Detector, balanced speed/accuracy

### Facial Landmarks
Uses OpenCV's Haar Cascade classifiers to detect:
- Eyes (using haarcascade_eye.xml)
- Face regions (using haarcascade_frontalface_default.xml)
- Mouth/smile regions (using haarcascade_smile.xml)

### ICAO Compliance
Evaluates photographs against ICAO Doc 9303 standards for machine-readable travel documents, ensuring photos meet international requirements for identification documents.

## Dependencies

Key libraries used:
- **DeepFace**: Face detection and analysis
- **OpenCV**: Image processing and facial landmark detection
- **Pillow (PIL)**: Image loading and manipulation
- **scikit-learn**: K-Means clustering for color extraction
- **pandas**: Data organization and table generation
- **TensorFlow/Keras**: Deep learning backend for DeepFace

## Troubleshooting

### Common Issues

1. **"No module named 'deepface'"**
   - Solution: Install dependencies with `uv pip install -r requirements.txt` (activate the venv first)

2. **First run is slow**
   - This is normal. DeepFace downloads models on first use (~100-200MB)

3. **"No face detected"**
   - Ensure the image contains a clear, frontal face
   - Try with better quality or higher resolution images
   - Check that the face is not too small in the image

4. **Memory errors**
   - Reduce image size before processing
   - Process images one at a time instead of in batch

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Disclaimer

This is a proof-of-concept tool for educational and testing purposes. For production use in official identity verification systems, please ensure compliance with local regulations and standards.

## Author

Flavio de Barros Vidal

## References

- [DeepFace GitHub Repository](https://github.com/serengil/deepface)
- [ICAO 9303 Standards](https://www.icao.int/publications/pages/publication.aspx?docnum=9303)
- Brazilian National ID (Carteira de Identidade Nacional - CIN) specifications
Prova de Conceito de avaliação da qualidade do CIN (parceria ITI e UnB)
