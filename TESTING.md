# Sample test images and usage examples

## Test Image Sources

For testing the ID card analyzer, you can use:

1. **Sample Brazilian ID Cards** (for testing purposes only)
   - Search for "Carteira de Identidade Nacional CIN exemplo" 
   - Use publicly available sample/template images
   - **Important**: Never use real ID cards without proper authorization

2. **Create Test Images**
   - Take a photo with proper lighting
   - Ensure face is frontal and clearly visible
   - Use minimum 600x400 pixel resolution
   - Follow ICAO guidelines for photo composition

## Quick Start Examples

### Example 1: Single Image Analysis
```bash
python id_card_analyzer.py test_images/sample_card.jpg
```

### Example 2: Multiple Images
```bash
python id_card_analyzer.py test_images/*.jpg
```

### Example 3: Different Image Formats
```bash
python id_card_analyzer.py card1.jpg card2.png card3.jpeg
```

## Expected Output Structure

After running the analyzer, you'll find:
- Console output with real-time progress
- Text file: `<image_name>_analysis.txt` for each image

## Testing Tips

1. **Good Quality Images**:
   - Clear, high-resolution photos
   - Frontal face view
   - Neutral expression
   - Good lighting without shadows
   - Plain background

2. **Testing Edge Cases**:
   - Low resolution images (to test resolution check)
   - Images with multiple faces
   - Profile or angled faces
   - Poor lighting conditions
   - Obscured facial features

3. **Performance Notes**:
   - First run downloads models (~100-200MB)
   - Subsequent runs are much faster
   - Processing time: ~5-15 seconds per image

## Sample Test Workflow

```bash
# 1. Setup environment
./setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Create test directory
mkdir -p test_images

# 4. Add your test images to test_images/

# 5. Run analyzer
python id_card_analyzer.py test_images/sample_card.jpg

# 6. Review results
cat sample_card_analysis.txt
```

## Interpreting Results

### Color Palette
- Shows the 5 most dominant colors
- Useful for detecting counterfeit or altered documents
- Brazilian ID cards typically have specific color schemes

### Face Detection
- Multiple detectors provide redundancy
- At least 2-3 detectors should confirm face presence
- Higher confidence scores indicate better detection

### Facial Landmarks
- Should detect at least 1 face and 2 eyes
- Mouth detection may vary based on image quality
- Nose detection uses approximation methods

### ICAO Compliance
- All checks should pass for compliant photos
- Failed checks indicate areas needing improvement
- Most critical: face detection, size, and positioning

## Common Test Scenarios

| Scenario | Expected Result |
|----------|----------------|
| High-quality frontal face | All checks pass |
| Low resolution (< 600x400) | Resolution check fails |
| Face too small | Face size ratio checks fail |
| Face off-center | Position checks fail |
| Multiple faces | Primary face detected, may affect accuracy |
| No face visible | Face detection fails, compliance fails |
| Over/underexposed | Brightness check fails |

## Privacy and Ethics

⚠️ **Important Reminders**:
- Never process real ID cards without authorization
- This is a proof-of-concept for educational purposes
- Respect privacy laws and regulations
- Do not store or share processed ID card images
- Obtain proper consent before analyzing any personal documents
