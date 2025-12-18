#!/usr/bin/env python3
"""
Integrated Meat Analysis System - FIXED VERSION
Combines TFLite freshness detection + advanced marbling analysis
Runs on Raspberry Pi with camera
"""

import os
import json
import time
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path

# TFLite for freshness
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# Image processing
import cv2


class IntegratedMeatAnalyzer:
    """Combined freshness + marbling analysis with improved logic"""
    
    def __init__(self, 
                 tflite_model_path='meat_freshness_model.tflite',
                 labels_path='labels.txt',
                 save_dir='analysis_results'):
        
        print("=" * 60)
        print("INTEGRATED MEAT ANALYSIS SYSTEM - OPTIMIZED")
        print("=" * 60)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize TFLite freshness detector
        print("\n[1/2] Loading TFLite freshness model...")
        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.freshness_labels = [line.strip() for line in f.readlines()]
        
        print(f"   Model loaded: {self.input_shape}")
        print(f"   Classes: {self.freshness_labels}")
        
        print("\n[2/2] Initializing advanced marbling analyzer...")
        print("   Using HSV + LAB color space analysis")
        print("   Smart fat detection with multiple algorithms")
        
        print("\n" + "=" * 60)
        print("SYSTEM READY")
        print("=" * 60 + "\n")
    
    def capture_image(self, preview_time=3):
        """Capture image from Pi Camera"""
        print(f"\nCapturing image (preview: {preview_time}s)...")
        
        try:
            from picamera2 import Picamera2
            camera = Picamera2()
            config = camera.create_still_configuration(main={"size": (1920, 1080)})
            camera.configure(config)
            camera.start()
            
            print(f"  Preview active...")
            time.sleep(preview_time)
            
            # Capture
            temp_path = 'temp_capture.jpg'
            camera.capture_file(temp_path)
            camera.stop()
            
            img = Image.open(temp_path)
            print(f"  Image captured: {img.size}")
            
            return np.array(img), temp_path
            
        except ImportError:
            print("  ERROR: picamera2 not found. Using test image if available.")
            if os.path.exists('test_image.jpg'):
                img = Image.open('test_image.jpg')
                return np.array(img), 'test_image.jpg'
            raise
    
    def analyze_freshness(self, image_array):
        """TFLite freshness detection"""
        print("\n" + "-" * 60)
        print("FRESHNESS ANALYSIS (TFLite)")
        print("-" * 60)
        
        # Preprocess
        img_resized = cv2.resize(image_array, 
                                (self.input_shape[2], self.input_shape[1]))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], img_batch)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        inference_time = (time.time() - start_time) * 1000
        
        # Results
        pred_idx = np.argmax(predictions)
        pred_class = self.freshness_labels[pred_idx]
        confidence = float(predictions[pred_idx])
        
        result = {
            'predicted_class': pred_class,
            'confidence': round(confidence, 4),
            'inference_time_ms': round(inference_time, 2),
            'all_probabilities': {
                self.freshness_labels[i]: round(float(predictions[i]), 4)
                for i in range(len(self.freshness_labels))
            }
        }
        
        print(f"\n  Predicted: {pred_class}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Inference time: {inference_time:.1f}ms")
        
        return result
    
    def analyze_marbling(self, image_array):
        """
        Advanced marbling quality analysis using HSV + LAB color spaces
        Mimics human visual assessment of meat marbling
        """
        print("\n" + "-" * 60)
        print("MARBLING ANALYSIS - ADVANCED")
        print("-" * 60)
        
        # Convert to different color spaces for robust analysis
        img_rgb = image_array.copy()
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        
        # Step 1: Segment meat region using multiple criteria
        print("\n  [1/5] Segmenting meat region...")
        meat_mask = self._segment_meat_advanced(img_rgb, img_hsv, img_lab)
        meat_pixels = np.sum(meat_mask)
        
        if meat_pixels < 1000:
            print("  WARNING: Very small meat region detected")
            return self._create_error_result("Insufficient meat detected in image")
        
        print(f"       Meat region: {meat_pixels:,} pixels")
        
        # Step 2: Detect fat using color thresholds
        print("  [2/5] Detecting fat pixels...")
        fat_mask = self._detect_fat_advanced(img_rgb, img_hsv, img_lab, meat_mask)
        fat_pixels = np.sum(fat_mask)
        fat_percentage = (fat_pixels / meat_pixels) * 100
        
        print(f"       Fat pixels: {fat_pixels:,}")
        print(f"       Fat percentage: {fat_percentage:.2f}%")
        
        # Step 3: Analyze fat distribution patterns
        print("  [3/5] Analyzing fat distribution...")
        distribution_info = self._analyze_fat_distribution(fat_mask, meat_mask)
        
        # Step 4: Detect marbling streaks
        print("  [4/5] Detecting marbling streaks...")
        streak_info = self._detect_marbling_streaks(fat_mask, meat_mask)
        
        print(f"       Marbling streaks: {streak_info['num_streaks']}")
        print(f"       Average streak size: {streak_info['avg_streak_size']:.1f} pixels")
        
        # Step 5: Final classification
        print("  [5/5] Classifying marbling quality...")
        classification = self._classify_marbling_quality(
            fat_percentage,
            distribution_info,
            streak_info
        )
        
        # Compile results
        result = {
            'label': classification['label'],
            'confidence': round(classification['confidence'], 4),
            'fat_percentage': round(fat_percentage, 2),
            'intramuscular_fat_percentage': round(distribution_info['internal_fat_pct'], 2),
            'subcutaneous_fat_percentage': round(distribution_info['outer_fat_pct'], 2),
            'num_marbling_streaks': streak_info['num_streaks'],
            'marbling_density': round(streak_info['marbling_density'], 4),
            'fat_distribution_score': round(distribution_info['distribution_score'], 4),
            'quality_grade': classification['quality_grade']
        }
        
        print(f"\n  Classification: {result['label']}")
        print(f"  Quality Grade: {result['quality_grade']}")
        print(f"  Fat content: {result['fat_percentage']:.2f}%")
        print(f"  Marbling density: {result['marbling_density']:.2f}")
        
        return result
    
    def _segment_meat_advanced(self, img_rgb, img_hsv, img_lab):
        """
        Advanced meat segmentation using multiple color space criteria
        Mimics human ability to identify meat vs background
        """
        h, w = img_rgb.shape[:2]
        meat_mask = np.zeros((h, w), dtype=bool)
        
        # Extract color channels
        r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
        h_channel, s_channel, v_channel = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
        l_channel, a_channel, b_channel = img_lab[:,:,0], img_lab[:,:,1], img_lab[:,:,2]
        
        # Criterion 1: Red meat has high R and positive a* (redness in LAB)
        red_meat_mask = (r > 80) & (a_channel > 128) & (s_channel > 20)
        
        # Criterion 2: HSV range for meat (red/pink hues)
        # Meat typically in red range: H=0-20 or H=160-180
        hue_mask = ((h_channel < 20) | (h_channel > 160)) & (s_channel > 25) & (v_channel > 40)
        
        # Criterion 3: Brightness range (not too dark, not too bright)
        brightness_mask = (v_channel > 30) & (v_channel < 240)
        
        # Combine criteria
        meat_mask = red_meat_mask | (hue_mask & brightness_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        meat_mask = cv2.morphologyEx(meat_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        meat_mask = cv2.morphologyEx(meat_mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only largest connected component (main meat piece)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(meat_mask, connectivity=8)
        if num_labels > 1:
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            meat_mask = (labels == largest_component).astype(np.uint8)
        
        return meat_mask.astype(bool)
    
    def _detect_fat_advanced(self, img_rgb, img_hsv, img_lab, meat_mask):
        """
        Advanced fat detection using multiple color characteristics
        Fat appears as white/cream colored regions in meat
        """
        h, w = img_rgb.shape[:2]
        fat_mask = np.zeros((h, w), dtype=bool)
        
        # Only consider pixels within meat region
        r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
        h_channel, s_channel, v_channel = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
        l_channel, a_channel, b_channel = img_lab[:,:,0], img_lab[:,:,1], img_lab[:,:,2]
        
        # Fat characteristics:
        # 1. High brightness (high V in HSV, high L in LAB)
        # 2. Low saturation (appears white/cream, not vivid red)
        # 3. Low a* channel in LAB (not red)
        # 4. RGB values are relatively balanced (white-ish)
        
        # Calculate relative brightness within meat
        meat_l_values = l_channel[meat_mask]
        if len(meat_l_values) > 0:
            l_threshold = np.percentile(meat_l_values, 70)  # Top 30% brightest
            
            # Fat detection criteria
            high_brightness = l_channel > l_threshold
            low_saturation = s_channel < 80
            low_redness = a_channel < 135  # Less red than typical meat
            high_value = v_channel > 120
            
            # RGB balance (fat is relatively balanced, not strongly red)
            rgb_std = np.std([r, g, b], axis=0)
            rgb_balanced = rgb_std < 40
            
            # Combine criteria
            fat_mask = meat_mask & high_brightness & low_saturation & low_redness & high_value & rgb_balanced
            
            # Additional threshold: significantly brighter than surrounding meat
            meat_median_l = np.median(meat_l_values)
            brightness_diff = l_channel > (meat_median_l + 15)
            fat_mask = fat_mask | (meat_mask & brightness_diff & low_saturation)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fat_mask = cv2.morphologyEx(fat_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Remove very small fat regions (likely noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fat_mask.astype(np.uint8), connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 20:  # Less than 20 pixels
                fat_mask[labels == i] = 0
        
        return fat_mask.astype(bool)
    
    def _analyze_fat_distribution(self, fat_mask, meat_mask):
        """
        Analyze how fat is distributed (outer rim vs internal marbling)
        """
        # Erode meat mask to define "internal" region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        internal_meat = cv2.erode(meat_mask.astype(np.uint8), kernel).astype(bool)
        outer_rim = meat_mask & ~internal_meat
        
        # Calculate fat in each region
        outer_fat = fat_mask & outer_rim
        internal_fat = fat_mask & internal_meat
        
        meat_pixels = np.sum(meat_mask)
        outer_fat_pixels = np.sum(outer_fat)
        internal_fat_pixels = np.sum(internal_fat)
        
        outer_fat_pct = (outer_fat_pixels / meat_pixels) * 100
        internal_fat_pct = (internal_fat_pixels / meat_pixels) * 100
        
        # Distribution score: higher for internal fat (marbling) vs outer fat
        # Good marbling has more internal fat
        if outer_fat_pixels + internal_fat_pixels > 0:
            distribution_score = internal_fat_pixels / (outer_fat_pixels + internal_fat_pixels)
        else:
            distribution_score = 0.0
        
        return {
            'outer_fat_pct': outer_fat_pct,
            'internal_fat_pct': internal_fat_pct,
            'distribution_score': distribution_score,
            'outer_fat_mask': outer_fat,
            'internal_fat_mask': internal_fat
        }
    
    def _detect_marbling_streaks(self, fat_mask, meat_mask):
        """
        Detect individual marbling streaks and analyze their characteristics
        Good marbling has many small-to-medium elongated fat streaks
        """
        # Find connected components (individual fat regions)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fat_mask.astype(np.uint8), connectivity=8
        )
        
        streaks = []
        total_streak_area = 0
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate aspect ratio (elongation)
            aspect_ratio = max(width, height) / (min(width, height) + 1)
            
            # Check if this is a marbling streak
            # Marbling characteristics:
            # - Not too small (> 30 pixels)
            # - Not too large (< 2% of meat area)
            # - Somewhat elongated (aspect ratio > 1.5)
            meat_area = np.sum(meat_mask)
            is_marbling = (
                area > 30 and 
                area < (meat_area * 0.02) and 
                aspect_ratio > 1.3
            )
            
            if is_marbling:
                streaks.append({
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'width': width,
                    'height': height
                })
                total_streak_area += area
        
        num_streaks = len(streaks)
        avg_streak_size = total_streak_area / num_streaks if num_streaks > 0 else 0
        
        # Marbling density: how much of the meat is fine marbling
        meat_area = np.sum(meat_mask)
        marbling_density = (total_streak_area / meat_area) if meat_area > 0 else 0
        
        return {
            'num_streaks': num_streaks,
            'avg_streak_size': avg_streak_size,
            'marbling_density': marbling_density,
            'streaks': streaks
        }
    
    def _classify_marbling_quality(self, fat_percentage, distribution_info, streak_info):
        """
        Classify meat marbling quality based on all analyzed features
        Uses industry-standard marbling assessment logic
        """
        num_streaks = streak_info['num_streaks']
        marbling_density = streak_info['marbling_density']
        distribution_score = distribution_info['distribution_score']
        internal_fat_pct = distribution_info['internal_fat_pct']
        
        # Classification logic based on USDA-style marbling grades
        
        # PRIME-grade marbling: abundant internal fat, many streaks
        if (internal_fat_pct > 8 and num_streaks > 20 and 
            marbling_density > 0.06 and distribution_score > 0.6):
            label = 'HIGHLY MARBLED'
            quality_grade = 'Prime'
            confidence = 0.90
        
        # CHOICE-grade marbling: moderate internal fat, good streaks
        elif (internal_fat_pct > 4 and num_streaks > 10 and 
              marbling_density > 0.03 and distribution_score > 0.5):
            label = 'WELL MARBLED'
            quality_grade = 'Choice'
            confidence = 0.85
        
        # SELECT-grade marbling: slight marbling
        elif (internal_fat_pct > 2 and num_streaks > 5 and 
              marbling_density > 0.015):
            label = 'MODERATELY MARBLED'
            quality_grade = 'Select'
            confidence = 0.80
        
        # Fatty but not marbled: high outer fat, low internal
        elif (fat_percentage > 15 and distribution_score < 0.3):
            label = 'FATTY (LOW MARBLING)'
            quality_grade = 'Standard'
            confidence = 0.82
        
        # Some fat but minimal marbling
        elif (fat_percentage > 5 and internal_fat_pct < 3):
            label = 'LIGHTLY MARBLED'
            quality_grade = 'Select'
            confidence = 0.75
        
        # Very lean meat
        elif fat_percentage < 3:
            label = 'LEAN'
            quality_grade = 'Select/Standard'
            confidence = 0.88
        
        # Default: some marbling present
        else:
            label = 'MODERATELY MARBLED'
            quality_grade = 'Select'
            confidence = 0.70
        
        return {
            'label': label,
            'quality_grade': quality_grade,
            'confidence': confidence
        }
    
    def _create_error_result(self, error_message):
        """Create error result when analysis fails"""
        return {
            'label': 'ANALYSIS FAILED',
            'confidence': 0.0,
            'fat_percentage': 0.0,
            'intramuscular_fat_percentage': 0.0,
            'subcutaneous_fat_percentage': 0.0,
            'num_marbling_streaks': 0,
            'marbling_density': 0.0,
            'fat_distribution_score': 0.0,
            'quality_grade': 'Unknown',
            'error': error_message
        }
    
    def run_complete_analysis(self, image_source=None):
        """Run complete integrated analysis"""
        print("\n" + "=" * 60)
        print("STARTING COMPLETE ANALYSIS")
        print("=" * 60)
        
        # Capture or load image
        if image_source is None:
            image_array, image_path = self.capture_image()
        elif isinstance(image_source, str):
            print(f"\nLoading image: {image_source}")
            image_array = np.array(Image.open(image_source))
            image_path = image_source
        else:
            image_array = image_source
            image_path = 'provided_array'
        
        # Ensure RGB format
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Save original
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_path = self.save_dir / f'original_{timestamp}.jpg'
        Image.fromarray(image_array.astype(np.uint8)).save(original_path)
        
        # Run analyses
        freshness_result = self.analyze_freshness(image_array)
        marbling_result = self.analyze_marbling(image_array)
        
        # Compile final results
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_path': str(original_path),
            'freshness_analysis': freshness_result,
            'marbling_analysis': marbling_result
        }
        
        # Save JSON results
        json_path = self.save_dir / f'results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {self.save_dir}")
        print(f"  - Original: {original_path.name}")
        print(f"  - JSON: {json_path.name}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Freshness: {freshness_result['predicted_class']} ({freshness_result['confidence']:.1%})")
        print(f"Marbling: {marbling_result['label']} - {marbling_result['quality_grade']}")
        print(f"Fat: {marbling_result['fat_percentage']:.1f}% ({marbling_result['intramuscular_fat_percentage']:.1f}% marbling)")
        print(f"Streaks: {marbling_result['num_marbling_streaks']}")
        print("=" * 60 + "\n")
        
        return results


if __name__ == "__main__":
    import sys
    
    # Initialize system
    analyzer = IntegratedMeatAnalyzer(
        tflite_model_path='meat_freshness_model.tflite',
        labels_path='labels.txt',
        save_dir='analysis_results'
    )
    
    # Run analysis
    if len(sys.argv) > 1:
        # Use provided image path
        results = analyzer.run_complete_analysis(sys.argv[1])
    else:
        # Capture from camera
        results = analyzer.run_complete_analysis()
    
    print(f"\nFull results JSON:\n{json.dumps(results, indent=2)}")