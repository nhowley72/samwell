import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from matplotlib.animation import FuncAnimation

# Path to the fine-tuned model (updated model)
MODEL_PATH = "finetuned_timesformer_middle50/final_model"
NUM_FRAMES = 8  # Updated to 8 frames
USE_MIDDLE_50_PERCENT = True  # Only use middle 50% of frames

# Load the processor and model
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForVideoClassification.from_pretrained(MODEL_PATH)

# Get class labels from directory names
DATASET_PATH = "badminton_dataset"
class_labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
id_to_label = {i: label for i, label in enumerate(class_labels)}

def predict_video(video_path, visualize=False):
    """
    Predict the class of a badminton video using the middle 50% of frames
    
    Args:
        video_path: Path to the video file
        visualize: Whether to read all frames for visualization
    """
    # Load video using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    all_frames = []  # Store all frames for visualization
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        print(f"Error: Video has no frames: {video_path}")
        return None
    
    # Calculate frame range for middle 50%
    if USE_MIDDLE_50_PERCENT:
        start_frame = int(frame_count * 0.25)
        end_frame = int(frame_count * 0.75)
        effective_frame_count = end_frame - start_frame
        print(f"Using frames {start_frame} to {end_frame} (middle 50% of {frame_count} frames)")
    else:
        start_frame = 0
        end_frame = frame_count
        effective_frame_count = frame_count
    
    # Sample frames uniformly for prediction
    if effective_frame_count >= NUM_FRAMES:
        if USE_MIDDLE_50_PERCENT:
            # Sample from the middle 50% of the video
            indices = np.linspace(start_frame, end_frame - 1, NUM_FRAMES, dtype=int)
        else:
            # Sample from the entire video
            indices = np.linspace(0, frame_count - 1, NUM_FRAMES, dtype=int)
    else:
        # If video is shorter, loop frames
        if USE_MIDDLE_50_PERCENT:
            indices = np.arange(start_frame, end_frame).repeat(NUM_FRAMES // effective_frame_count + 1)[:NUM_FRAMES]
        else:
            indices = np.arange(frame_count).repeat(NUM_FRAMES // frame_count + 1)[:NUM_FRAMES]
    
    # Read all frames for visualization
    if visualize:
        for i in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Mark frames used for prediction
                if USE_MIDDLE_50_PERCENT and start_frame <= i < end_frame:
                    # Add a green border to frames in the middle 50%
                    border_size = 3
                    frame_rgb[:border_size, :, :] = [0, 255, 0]  # Top border
                    frame_rgb[-border_size:, :, :] = [0, 255, 0]  # Bottom border
                    frame_rgb[:, :border_size, :] = [0, 255, 0]  # Left border
                    frame_rgb[:, -border_size:, :] = [0, 255, 0]  # Right border
                
                all_frames.append(frame_rgb)
    
    # Reset the video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Read selected frames for prediction
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Check if we have enough frames
    if len(frames) < NUM_FRAMES:
        # If we don't have enough frames, duplicate the last frame
        last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
        while len(frames) < NUM_FRAMES:
            frames.append(last_frame)
    
    # Process frames with the processor
    inputs = processor(frames, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get all class probabilities
        probabilities = torch.softmax(logits, dim=1)[0]
        
        # Get top prediction
        predicted_class_idx = logits.argmax(-1).item()
    
    # Get the predicted class label
    predicted_label = id_to_label[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    
    # Create dictionary of all class probabilities
    all_probs = {id_to_label[i]: prob.item() for i, prob in enumerate(probabilities)}
    
    result = {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "class_id": predicted_class_idx,
        "all_probabilities": all_probs,
        "frames": frames,
        "all_frames": all_frames if visualize else None,
        "sampled_indices": indices,  # Store which frames were used for prediction
        "middle_range": (start_frame, end_frame) if USE_MIDDLE_50_PERCENT else (0, frame_count)
    }
    
    return result

def visualize_prediction(result, true_class, video_path):
    """
    Visualize the prediction results with a bar chart of probabilities
    and display the sampled frames
    """
    if not result:
        return
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(15, 12))
    
    # Plot the probabilities as a bar chart
    ax1 = fig.add_subplot(2, 1, 1)
    probs = result["all_probabilities"]
    classes = list(probs.keys())
    values = list(probs.values())
    
    # Sort by probability
    sorted_indices = np.argsort(values)[::-1]
    classes = [classes[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    bars = ax1.bar(classes, values, color='skyblue')
    
    # Highlight the true class and predicted class
    for i, cls in enumerate(classes):
        if cls == true_class:
            bars[i].set_color('green')
        elif cls == result["predicted_class"] and cls != true_class:
            bars[i].set_color('red')
    
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Class Probabilities - True: {true_class}, Predicted: {result["predicted_class"]} ({result["confidence"]:.4f})')
    plt.xticks(rotation=45, ha='right')
    
    # Plot the sampled frames
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis('off')
    
    # Create a grid of the sampled frames
    frames = result["frames"]
    grid_size = int(np.ceil(np.sqrt(len(frames))))
    grid = np.zeros((grid_size * frames[0].shape[0], grid_size * frames[0].shape[1], 3), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        row = i // grid_size
        col = i % grid_size
        h, w = frame.shape[:2]
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame
    
    ax2.imshow(grid)
    
    # Show which frames were used
    start_frame, end_frame = result["middle_range"]
    frame_indices = result["sampled_indices"]
    ax2.set_title(f'Sampled {NUM_FRAMES} Frames from {os.path.basename(video_path)}\nUsing middle 50% (frames {start_frame}-{end_frame})\nIndices: {frame_indices}')
    
    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(os.path.basename(video_path))[0]}_middle50_prediction.png")
    plt.show()

def create_video_with_prediction(result, true_class, video_path):
    """
    Create a video with prediction overlay
    """
    if not result or not result["all_frames"]:
        return
    
    all_frames = result["all_frames"]
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    all_probs = result["all_probabilities"]
    start_frame, end_frame = result["middle_range"]
    
    # Create output video
    output_path = f"{os.path.splitext(os.path.basename(video_path))[0]}_middle50_annotated.mp4"
    
    # Get video dimensions
    height, width = all_frames[0].shape[:2]
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    # Add prediction overlay to each frame
    for i, frame in enumerate(all_frames):
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add prediction text
        cv2.putText(frame_bgr, f"True: {true_class}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Color based on correctness
        color = (0, 255, 0) if predicted_class == true_class else (0, 0, 255)
        cv2.putText(frame_bgr, f"Pred: {predicted_class} ({confidence:.2f})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add top 3 predictions
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for j, (cls, prob) in enumerate(sorted_probs[:3]):
            cv2.putText(frame_bgr, f"{j+1}. {cls}: {prob:.2f}", 
                       (10, 90 + j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame info
        in_middle = start_frame <= i < end_frame
        frame_status = "USED" if in_middle else "IGNORED"
        status_color = (0, 255, 0) if in_middle else (0, 0, 255)  # Green if used, red if ignored
        
        cv2.putText(frame_bgr, f"Frame {i}/{len(all_frames)} - {frame_status}", 
                   (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Add info about using 8 frames and middle 50%
        cv2.putText(frame_bgr, f"Model: 8 frames, middle 50%", 
                   (width - 250, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame_bgr)
    
    out.release()
    print(f"Annotated video saved to {output_path}")
    return output_path

def main():
    print("Testing the model trained with 8 frames from the middle 50% of videos")
    print("Available classes:", class_labels)
    
    # Test on one video from each class
    for class_name in class_labels:
        class_dir = os.path.join(DATASET_PATH, class_name)
        videos = [f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if videos:
            # Get the last video (to test on one we likely didn't train on)
            test_video = videos[-1]
            video_path = os.path.join(class_dir, test_video)
            
            print(f"\nTesting video: {video_path}")
            # Use middle_only=True to only consider the middle 50% of frames
            result = predict_video(video_path, visualize=True)
            
            if result:
                print(f"True class: {class_name}")
                print(f"Predicted class: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Correct: {'✓' if result['predicted_class'] == class_name else '✗'}")
                
                # Print all class probabilities
                print("\nAll class probabilities:")
                for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {cls}: {prob:.4f}")
                
                # Visualize the prediction
                visualize_prediction(result, class_name, video_path)
                
                # Create annotated video
                create_video_with_prediction(result, class_name, video_path)

if __name__ == "__main__":
    main() 