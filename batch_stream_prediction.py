import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random

# Path to the fine-tuned model (updated model)
MODEL_PATH = "finetuned_timesformer_middle50/final_model"
NUM_FRAMES = 8  # Updated to 8 frames
USE_MIDDLE_50_PERCENT = True  # Only use middle 50% of frames

# Video playback settings
OUTPUT_VIDEO_PATH = "combined_stream_with_predictions.mp4"
FRAME_RATE = 30
DISPLAY_SCALE = 1.0  # Scale factor for display window

# Load the processor and model
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForVideoClassification.from_pretrained(MODEL_PATH)

# Get class labels from directory names
DATASET_PATH = "badminton_dataset"
class_labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
label_to_id = {label: i for i, label in enumerate(class_labels)}
id_to_label = {i: label for i, label in enumerate(class_labels)}

def collect_all_videos():
    """Collect all videos from the dataset"""
    videos_with_labels = []
    
    for class_name in class_labels:
        class_dir = os.path.join(DATASET_PATH, class_name)
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(class_dir, filename)
                videos_with_labels.append((video_path, class_name))
    
    return videos_with_labels

def process_frame_for_prediction(frame, frames_buffer, current_frame_index, total_frames, start_frame, end_frame):
    """Process a frame for prediction and add it to the buffer if needed"""
    # Convert BGR to RGB for model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # If within sampling range, check if this frame should be sampled
    if start_frame <= current_frame_index < end_frame:
        # Mark as a frame from the middle 50%
        # Add green border
        border_size = 3
        frame_rgb[:border_size, :, :] = [0, 255, 0]
        frame_rgb[-border_size:, :, :] = [0, 255, 0]
        frame_rgb[:, :border_size, :] = [0, 255, 0]
        frame_rgb[:, -border_size:, :] = [0, 255, 0]
        
        # Add to buffer if it's one of the frames we need
        frames_buffer.append(frame_rgb)
        
    return frame_rgb

def get_predictions_for_video_segment(frames_buffer, all_frame_indices):
    """Get predictions for a segment of frames"""
    if len(frames_buffer) < NUM_FRAMES:
        # If not enough frames, duplicate the last frame
        last_frame = frames_buffer[-1] if frames_buffer else np.zeros((224, 224, 3), dtype=np.uint8)
        while len(frames_buffer) < NUM_FRAMES:
            frames_buffer.append(last_frame)
    
    # Use only the required number of frames
    sample_frames = frames_buffer[:NUM_FRAMES]
    
    # Process frames with the model
    try:
        inputs = processor(sample_frames, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            
            # Get top prediction
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = id_to_label[predicted_class_idx]
            confidence = probabilities[predicted_class_idx].item()
            
            # Create dictionary of all class probabilities
            all_probs = {id_to_label[i]: prob.item() for i, prob in enumerate(probabilities)}
            
            return predicted_label, confidence, all_probs
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return "Error", 0.0, {label: 0.0 for label in class_labels}

def add_prediction_overlay(frame, true_class, predicted_class, confidence, all_probs, frame_info):
    """Add prediction overlay to a frame"""
    height, width = frame.shape[:2]
    
    # Create a copy of the frame
    overlay = frame.copy()
    
    # Add semi-transparent overlay for better text visibility
    overlay_alpha = 0.3
    cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
    cv2.rectangle(overlay, (width-280, 0), (width, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0, frame)
    
    # Add true class and prediction
    cv2.putText(frame, f"True: {true_class}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Color based on correctness
    color = (0, 255, 0) if predicted_class == true_class else (0, 0, 255)
    cv2.putText(frame, f"Pred: {predicted_class} ({confidence:.2f})", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add top 3 predictions
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (cls, prob) in enumerate(sorted_probs[:3]):
        cv2.putText(frame, f"{i+1}. {cls}: {prob:.2f}", 
                   (10, 90 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add frame info
    cv2.putText(frame, frame_info, 
               (width - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def create_combined_video_stream(videos_with_labels, output_path, display=True):
    """Create a continuous stream of videos with real-time predictions"""
    # Shuffle videos for a more diverse stream
    random.shuffle(videos_with_labels)
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    # Get first video to determine dimensions
    cap = cv2.VideoCapture(videos_with_labels[0][0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FRAME_RATE, (width, height))
    
    # Create window if display is True
    if display:
        cv2.namedWindow("Badminton Shot Classification Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Badminton Shot Classification Stream", 
                        int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE))
    
    # Process each video
    for video_index, (video_path, true_class) in enumerate(videos_with_labels):
        print(f"\nProcessing video {video_index+1}/{len(videos_with_labels)}: {os.path.basename(video_path)}")
        print(f"True class: {true_class}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            continue
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range for middle 50%
        if USE_MIDDLE_50_PERCENT:
            start_frame = int(frame_count * 0.25)
            end_frame = int(frame_count * 0.75)
        else:
            start_frame = 0
            end_frame = frame_count
        
        # Calculate frames to sample
        if (end_frame - start_frame) >= NUM_FRAMES:
            sample_indices = np.linspace(start_frame, end_frame - 1, NUM_FRAMES, dtype=int)
        else:
            sample_indices = np.arange(start_frame, end_frame).repeat(NUM_FRAMES // (end_frame - start_frame) + 1)[:NUM_FRAMES]
        
        print(f"Using frames {start_frame} to {end_frame} (middle 50% of {frame_count} frames)")
        print(f"Sampling frames at indices: {sample_indices}")
        
        # Initialize variables for processing
        frames_buffer = []
        last_prediction = None
        last_confidence = 0.0
        last_all_probs = {label: 0.0 for label in class_labels}
        current_frame_index = 0
        
        # Add title frame
        title_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(title_frame, f"Video: {os.path.basename(video_path)}", 
                   (width//2 - 200, height//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(title_frame, f"Class: {true_class}", 
                   (width//2 - 150, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Display title frame for 2 seconds
        for _ in range(int(FRAME_RATE * 2)):
            out.write(title_frame)
            if display:
                cv2.imshow("Badminton Shot Classification Stream", title_frame)
                if cv2.waitKey(int(1000/FRAME_RATE)) & 0xFF == ord('q'):
                    break
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_rgb = process_frame_for_prediction(
                frame, frames_buffer, current_frame_index, 
                frame_count, start_frame, end_frame
            )
            
            # Get predictions at appropriate times
            if current_frame_index in sample_indices:
                if len(frames_buffer) >= NUM_FRAMES:
                    predicted_label, confidence, all_probs = get_predictions_for_video_segment(
                        frames_buffer, sample_indices
                    )
                    last_prediction = predicted_label
                    last_confidence = confidence
                    last_all_probs = all_probs
            
            # Convert back to BGR for OpenCV
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Add frame info
            frame_info = f"Frame {current_frame_index}/{frame_count}"
            if start_frame <= current_frame_index < end_frame:
                frame_info += " - USED"
            else:
                frame_info += " - IGNORED"
            
            # Add prediction overlay
            if last_prediction:
                display_frame = add_prediction_overlay(
                    display_frame, true_class, last_prediction, 
                    last_confidence, last_all_probs, frame_info
                )
                
                # Record prediction for evaluation (only once per video)
                if current_frame_index == end_frame - 1:
                    all_predictions.append(last_prediction)
                    all_true_labels.append(true_class)
                    all_confidences.append(last_confidence)
            
            # Write frame to output video
            out.write(display_frame)
            
            # Display frame if requested
            if display:
                cv2.imshow("Badminton Shot Classification Stream", display_frame)
                if cv2.waitKey(int(1000/FRAME_RATE)) & 0xFF == ord('q'):
                    break
            
            current_frame_index += 1
        
        # Add a black frame transition
        transition_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(int(FRAME_RATE * 0.5)):  # Half-second transition
            out.write(transition_frame)
            if display:
                cv2.imshow("Badminton Shot Classification Stream", transition_frame)
                if cv2.waitKey(int(1000/FRAME_RATE)) & 0xFF == ord('q'):
                    break
        
        # Release video capture
        cap.release()
    
    # Display evaluation results
    print("\nEvaluation complete!")
    print(f"Processed {len(videos_with_labels)} videos")
    
    # Create evaluation summary
    evaluation = {
        "predictions": all_predictions,
        "true_labels": all_true_labels,
        "confidences": all_confidences
    }
    
    # Add evaluation summary to video
    create_evaluation_summary(evaluation, out, width, height, display)
    
    # Release video writer and destroy windows
    out.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"Combined video stream saved to: {output_path}")
    return evaluation

def create_evaluation_summary(evaluation, video_writer, width, height, display=True):
    """Create and add evaluation summary frames to the video"""
    # Get data
    predictions = evaluation["predictions"]
    true_labels = evaluation["true_labels"]
    confidences = evaluation["confidences"]
    
    # Calculate metrics
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(predictions) if predictions else 0
    
    # Create summary frame
    summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(summary_frame, "EVALUATION SUMMARY", 
               (width//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Add accuracy
    cv2.putText(summary_frame, f"Overall Accuracy: {accuracy:.2f} ({correct}/{len(predictions)})", 
               (width//2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Calculate class-wise accuracy
    class_correct = {label: 0 for label in class_labels}
    class_total = {label: 0 for label in class_labels}
    
    for pred, true in zip(predictions, true_labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    
    # Add class-wise accuracy
    y_pos = 150
    for label in class_labels:
        acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
        cv2.putText(summary_frame, f"{label}: {acc:.2f} ({class_correct[label]}/{class_total[label]})", 
                   (width//2 - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 40
    
    # Display for 5 seconds
    for _ in range(int(FRAME_RATE * 5)):
        video_writer.write(summary_frame)
        if display:
            cv2.imshow("Badminton Shot Classification Stream", summary_frame)
            if cv2.waitKey(int(1000/FRAME_RATE)) & 0xFF == ord('q'):
                break
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=class_labels)
    
    # Create a matplotlib figure with the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the figure
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    # Read the saved confusion matrix image
    cm_img = cv2.imread(cm_path)
    cm_img = cv2.resize(cm_img, (width, height))
    
    # Add title
    cv2.putText(cm_img, "CONFUSION MATRIX", 
               (width//2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Display for 5 seconds
    for _ in range(int(FRAME_RATE * 5)):
        video_writer.write(cm_img)
        if display:
            cv2.imshow("Badminton Shot Classification Stream", cm_img)
            if cv2.waitKey(int(1000/FRAME_RATE)) & 0xFF == ord('q'):
                break

def print_evaluation_report(evaluation):
    """Print detailed evaluation report"""
    predictions = evaluation["predictions"]
    true_labels = evaluation["true_labels"]
    
    print("\n==== CLASSIFICATION REPORT ====")
    print(classification_report(true_labels, predictions, target_names=class_labels))
    
    print("\n==== CONFUSION MATRIX ====")
    cm = confusion_matrix(true_labels, predictions, labels=class_labels)
    print(cm)
    
    # Create a nice confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to: confusion_matrix.png")

if __name__ == "__main__":
    print("Creating continuous stream of badminton videos with real-time predictions")
    print("Model: Using 8 frames from middle 50% of each video")
    print("Class labels:", class_labels)
    
    # Collect all videos
    videos_with_labels = collect_all_videos()
    print(f"Found {len(videos_with_labels)} videos")
    
    # Create combined video stream
    evaluation = create_combined_video_stream(videos_with_labels, OUTPUT_VIDEO_PATH, display=True)
    
    # Print detailed evaluation report
    print_evaluation_report(evaluation) 