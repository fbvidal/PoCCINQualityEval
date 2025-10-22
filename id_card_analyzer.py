#!/usr/bin/env python3
"""
Brazilian National ID Card Analyzer
Analyzes ID card images for color palette, facial features, and ICAO 9303 compliance
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from sklearn.cluster import KMeans
import pandas as pd
from deepface import DeepFace
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class ColorPaletteExtractor:
    """Extract dominant colors from an image"""
    
    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors
    
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def extract_palette(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract dominant colors from image
        Returns list of dicts with hex codes and percentages
        """
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        # Reshape to 2D array of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and their counts
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        # Build palette with percentages
        palette = []
        for i in range(self.n_colors):
            color_rgb = colors[i]
            hex_code = self.rgb_to_hex(color_rgb)
            percentage = (label_counts[i] / total_pixels) * 100
            palette.append({
                'hex': hex_code,
                'rgb': tuple(color_rgb.astype(int)),
                'percentage': round(percentage, 2)
            })
        
        # Sort by percentage (descending)
        palette.sort(key=lambda x: x['percentage'], reverse=True)
        
        return palette


class FaceAnalyzer:
    """Analyze facial features using DeepFace"""
    
    def __init__(self):
        self.detectors = ['opencv', 'retinaface', 'mtcnn', 'ssd']
    
    def detect_face(self, image_path: str, detector: str = 'opencv') -> Dict[str, Any]:
        """
        Detect face and extract facial landmarks
        """
        try:
            # Detect face
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector,
                enforce_detection=False
            )
            
            if not face_objs or len(face_objs) == 0:
                return {
                    'detected': False,
                    'detector': detector,
                    'error': 'No face detected'
                }
            
            face_obj = face_objs[0]
            
            # Get face region
            facial_area = face_obj.get('facial_area', {})
            
            return {
                'detected': True,
                'detector': detector,
                'confidence': face_obj.get('confidence', 0),
                'facial_area': facial_area,
                'face_coordinates': {
                    'x': facial_area.get('x', 0),
                    'y': facial_area.get('y', 0),
                    'w': facial_area.get('w', 0),
                    'h': facial_area.get('h', 0)
                }
            }
        
        except Exception as e:
            return {
                'detected': False,
                'detector': detector,
                'error': str(e)
            }
    
    def analyze_with_multiple_detectors(self, image_path: str) -> Dict[str, Any]:
        """Run detection with multiple detectors"""
        results = {}
        
        for detector in self.detectors:
            print(f"  Running {detector} detector...")
            result = self.detect_face(image_path, detector)
            results[detector] = result
        
        return results
    
    def extract_facial_landmarks(self, image_path: str) -> Dict[str, Any]:
        """
        Extract facial landmarks (eyes, nose, mouth, etc.)
        Using OpenCV's cascade classifiers for detailed detection
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        landmarks = {
            'eyes': [],
            'face': [],
            'mouth': [],
            'nose': []
        }
        
        # Load cascade classifiers
        try:
            # Face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            landmarks['face'] = [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} 
                                 for x, y, w, h in faces]
            
            # Eye detection
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            landmarks['eyes'] = [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} 
                                for x, y, w, h in eyes]
            
            # Nose detection (using profile cascade as approximation)
            nose_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Mouth detection (smile cascade as proxy)
            smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                landmarks['mouth'] = [{'x': int(fx+x), 'y': int(fy+y), 'w': int(w), 'h': int(h)} 
                                     for x, y, w, h in smiles]
            
        except Exception as e:
            print(f"  Warning: Error in landmark detection: {str(e)}")
        
        return landmarks


class ICAOComplianceEvaluator:
    """Evaluate ID card image compliance with ICAO 9303 standards"""
    
    def __init__(self):
        # ICAO 9303 standard requirements (simplified)
        self.standards = {
            'face_width_ratio': (0.50, 0.75),  # Face should be 50-75% of image width
            'face_height_ratio': (0.60, 0.90),  # Face should be 60-90% of image height
            'face_position_x': (0.35, 0.65),  # Face center should be centered horizontally
            'face_position_y': (0.30, 0.60),  # Face center in upper-middle area
            'min_resolution': (600, 400),  # Minimum resolution (width, height)
            'max_brightness': 240,  # Maximum average brightness
            'min_brightness': 50,  # Minimum average brightness
            'max_contrast_std': 70,  # Maximum standard deviation for contrast
        }
    
    def evaluate_compliance(self, image_path: str, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate image compliance with ICAO 9303 standards
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not read image'}
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = {}
        
        # Check resolution
        results['resolution'] = {
            'value': f"{width}x{height}",
            'compliant': width >= self.standards['min_resolution'][0] and 
                        height >= self.standards['min_resolution'][1],
            'standard': f"Min {self.standards['min_resolution'][0]}x{self.standards['min_resolution'][1]}"
        }
        
        # Check brightness
        avg_brightness = np.mean(gray)
        results['brightness'] = {
            'value': round(avg_brightness, 2),
            'compliant': self.standards['min_brightness'] <= avg_brightness <= self.standards['max_brightness'],
            'standard': f"{self.standards['min_brightness']}-{self.standards['max_brightness']}"
        }
        
        # Check contrast
        contrast_std = np.std(gray)
        results['contrast'] = {
            'value': round(contrast_std, 2),
            'compliant': contrast_std <= self.standards['max_contrast_std'],
            'standard': f"Max {self.standards['max_contrast_std']}"
        }
        
        # Face-related checks (if face detected)
        if face_data and face_data.get('detected'):
            coords = face_data.get('face_coordinates', {})
            face_w = coords.get('w', 0)
            face_h = coords.get('h', 0)
            face_x = coords.get('x', 0)
            face_y = coords.get('y', 0)
            
            if face_w > 0 and face_h > 0:
                # Face size ratios
                face_width_ratio = face_w / width
                face_height_ratio = face_h / height
                
                results['face_width_ratio'] = {
                    'value': round(face_width_ratio, 3),
                    'compliant': self.standards['face_width_ratio'][0] <= face_width_ratio <= 
                                self.standards['face_width_ratio'][1],
                    'standard': f"{self.standards['face_width_ratio'][0]}-{self.standards['face_width_ratio'][1]}"
                }
                
                results['face_height_ratio'] = {
                    'value': round(face_height_ratio, 3),
                    'compliant': self.standards['face_height_ratio'][0] <= face_height_ratio <= 
                                self.standards['face_height_ratio'][1],
                    'standard': f"{self.standards['face_height_ratio'][0]}-{self.standards['face_height_ratio'][1]}"
                }
                
                # Face position (center of face)
                face_center_x = (face_x + face_w / 2) / width
                face_center_y = (face_y + face_h / 2) / height
                
                results['face_position_x'] = {
                    'value': round(face_center_x, 3),
                    'compliant': self.standards['face_position_x'][0] <= face_center_x <= 
                                self.standards['face_position_x'][1],
                    'standard': f"{self.standards['face_position_x'][0]}-{self.standards['face_position_x'][1]}"
                }
                
                results['face_position_y'] = {
                    'value': round(face_center_y, 3),
                    'compliant': self.standards['face_position_y'][0] <= face_center_y <= 
                                self.standards['face_position_y'][1],
                    'standard': f"{self.standards['face_position_y'][0]}-{self.standards['face_position_y'][1]}"
                }
        else:
            results['face_detection'] = {
                'value': 'No face detected',
                'compliant': False,
                'standard': 'Face must be detected'
            }
        
        return results


class IDCardAnalyzer:
    """Main analyzer class that orchestrates all analysis"""
    
    def __init__(self):
        self.color_extractor = ColorPaletteExtractor(n_colors=5)
        self.face_analyzer = FaceAnalyzer()
        self.compliance_evaluator = ICAOComplianceEvaluator()
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform complete analysis of ID card image
        """
        if not os.path.exists(image_path):
            return {'error': f'Image file not found: {image_path}'}
        
        print(f"\nAnalyzing: {image_path}")
        print("=" * 80)
        
        results = {
            'image_path': image_path,
            'color_palette': None,
            'face_detection': None,
            'facial_landmarks': None,
            'icao_compliance': None
        }
        
        # Extract color palette
        print("\n1. Extracting color palette...")
        try:
            results['color_palette'] = self.color_extractor.extract_palette(image_path)
            print(f"   ✓ Extracted {len(results['color_palette'])} dominant colors")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            results['color_palette'] = {'error': str(e)}
        
        # Detect faces with multiple detectors
        print("\n2. Detecting faces with multiple detectors...")
        try:
            results['face_detection'] = self.face_analyzer.analyze_with_multiple_detectors(image_path)
            detected_count = sum(1 for r in results['face_detection'].values() if r.get('detected'))
            print(f"   ✓ Face detected by {detected_count}/{len(self.face_analyzer.detectors)} detectors")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            results['face_detection'] = {'error': str(e)}
        
        # Extract facial landmarks
        print("\n3. Extracting facial landmarks...")
        try:
            results['facial_landmarks'] = self.face_analyzer.extract_facial_landmarks(image_path)
            landmark_counts = {k: len(v) for k, v in results['facial_landmarks'].items()}
            print(f"   ✓ Found landmarks: {landmark_counts}")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            results['facial_landmarks'] = {'error': str(e)}
        
        # Evaluate ICAO compliance (use best face detection result)
        print("\n4. Evaluating ICAO 9303 compliance...")
        try:
            best_face = self._get_best_face_detection(results['face_detection'])
            results['icao_compliance'] = self.compliance_evaluator.evaluate_compliance(
                image_path, best_face
            )
            compliant_count = sum(1 for v in results['icao_compliance'].values() 
                                if isinstance(v, dict) and v.get('compliant'))
            total_checks = len(results['icao_compliance'])
            print(f"   ✓ Compliance: {compliant_count}/{total_checks} checks passed")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            results['icao_compliance'] = {'error': str(e)}
        
        return results
    
    def _get_best_face_detection(self, face_detections: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best face detection result from multiple detectors"""
        if not face_detections or 'error' in face_detections:
            return {}
        
        # Prefer detections with higher confidence
        best = None
        best_confidence = -1
        
        for detector, result in face_detections.items():
            if result.get('detected'):
                confidence = result.get('confidence', 0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best = result
        
        return best or {}
    
    def generate_report_table(self, results: Dict[str, Any]) -> str:
        """
        Generate formatted table report
        """
        report_lines = []
        report_lines.append("\n" + "=" * 80)
        report_lines.append("BRAZILIAN NATIONAL ID CARD ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nImage: {results['image_path']}")
        
        # Color Palette Section
        report_lines.append("\n" + "-" * 80)
        report_lines.append("COLOR PALETTE")
        report_lines.append("-" * 80)
        
        if results['color_palette'] and 'error' not in results['color_palette']:
            color_data = []
            for i, color in enumerate(results['color_palette'], 1):
                color_data.append({
                    'Rank': i,
                    'Hex Code': color['hex'],
                    'RGB': str(color['rgb']),
                    'Percentage': f"{color['percentage']:.2f}%"
                })
            df_colors = pd.DataFrame(color_data)
            report_lines.append(df_colors.to_string(index=False))
        else:
            report_lines.append("Error extracting color palette")
        
        # Face Detection Section
        report_lines.append("\n" + "-" * 80)
        report_lines.append("FACE DETECTION RESULTS")
        report_lines.append("-" * 80)
        
        if results['face_detection'] and 'error' not in results['face_detection']:
            face_data = []
            for detector, result in results['face_detection'].items():
                face_data.append({
                    'Detector': detector.upper(),
                    'Detected': '✓' if result.get('detected') else '✗',
                    'Confidence': f"{result.get('confidence', 0):.3f}" if result.get('detected') else 'N/A'
                })
            df_faces = pd.DataFrame(face_data)
            report_lines.append(df_faces.to_string(index=False))
        else:
            report_lines.append("Error in face detection")
        
        # Facial Landmarks Section
        report_lines.append("\n" + "-" * 80)
        report_lines.append("FACIAL LANDMARKS")
        report_lines.append("-" * 80)
        
        if results['facial_landmarks'] and 'error' not in results['facial_landmarks']:
            landmarks_data = []
            for feature, locations in results['facial_landmarks'].items():
                landmarks_data.append({
                    'Feature': feature.capitalize(),
                    'Count': len(locations),
                    'Detected': '✓' if len(locations) > 0 else '✗'
                })
            df_landmarks = pd.DataFrame(landmarks_data)
            report_lines.append(df_landmarks.to_string(index=False))
        else:
            report_lines.append("Error extracting facial landmarks")
        
        # ICAO Compliance Section
        report_lines.append("\n" + "-" * 80)
        report_lines.append("ICAO 9303 COMPLIANCE EVALUATION")
        report_lines.append("-" * 80)
        
        if results['icao_compliance'] and 'error' not in results['icao_compliance']:
            compliance_data = []
            for criterion, data in results['icao_compliance'].items():
                if isinstance(data, dict):
                    compliance_data.append({
                        'Criterion': criterion.replace('_', ' ').title(),
                        'Value': str(data.get('value', 'N/A')),
                        'Standard': data.get('standard', 'N/A'),
                        'Result': '✓ PASS' if data.get('compliant') else '✗ FAIL'
                    })
            df_compliance = pd.DataFrame(compliance_data)
            report_lines.append(df_compliance.to_string(index=False))
            
            # Summary
            total_checks = len(compliance_data)
            passed_checks = sum(1 for item in compliance_data if '✓' in item['Result'])
            report_lines.append(f"\nCompliance Summary: {passed_checks}/{total_checks} checks passed")
        else:
            report_lines.append("Error evaluating ICAO compliance")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python id_card_analyzer.py <image_path> [image_path2 ...]")
        print("\nExample:")
        print("  python id_card_analyzer.py sample_id_card.jpg")
        print("  python id_card_analyzer.py card1.jpg card2.png card3.jpg")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    analyzer = IDCardAnalyzer()
    
    # Analyze each image
    all_results = []
    for image_path in image_paths:
        results = analyzer.analyze(image_path)
        all_results.append(results)
        
        # Generate and print report
        report = analyzer.generate_report_table(results)
        print(report)
        
        # Save report to file
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_analysis.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✓ Report saved to: {output_filename}\n")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Processed {len(all_results)} image(s).")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
