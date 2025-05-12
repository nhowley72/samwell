import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor, 
    AutoModelForVideoClassification,
    TrainingArguments, 
    Trainer
)
import cv2
from sklearn.model_selection import train_test_split
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define paths and parameters
DATASET_PATH = "badminton_dataset"
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
OUTPUT_DIR = "finetuned_timesformer_middle50"  # New output directory to preserve the old model
NUM_FRAMES = 8  # Increased from 4 to 8 frames
BATCH_SIZE = 1  # Keep batch size at 1 to avoid memory issues
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3  # Keep at 3 epochs
USE_MIDDLE_50_PERCENT = True  # Only use middle 50% of frames

# Get class labels from directory names
class_labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
label_to_id = {label: i for i, label in enumerate(class_labels)}
id_to_label = {i: label for i, label in enumerate(class_labels)}
num_labels = len(class_labels)

print(f"Found {num_labels} classes: {class_labels}")

# Initialize the processor
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

class BadmintonVideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video using OpenCV
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count <= 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # Calculate frame range for middle 50%
            if USE_MIDDLE_50_PERCENT:
                start_frame = int(frame_count * 0.25)
                end_frame = int(frame_count * 0.75)
                effective_frame_count = end_frame - start_frame
            else:
                start_frame = 0
                end_frame = frame_count
                effective_frame_count = frame_count
                
            # Sample frames uniformly
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
            
            # Read selected frames
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
            
            # Process frames with the TimeSformer processor
            inputs = self.processor(
                frames,  # List of frames
                return_tensors="pt",
            )
            
            # Add label
            inputs["labels"] = torch.tensor(label)
            
            return inputs
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return a placeholder with the correct structure
            # This is a fallback to avoid breaking the training loop
            dummy_inputs = {
                "pixel_values": torch.zeros((1, NUM_FRAMES, 3, 224, 224)),
                "labels": torch.tensor(label)
            }
            return dummy_inputs

def collect_videos():
    video_paths = []
    labels = []
    
    for class_name in class_labels:
        class_dir = os.path.join(DATASET_PATH, class_name)
        class_id = label_to_id[class_name]
        
        # Limit to 10 videos per class to reduce dataset size even further
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        files = files[:10] if len(files) > 10 else files
        
        for filename in files:
            video_path = os.path.join(class_dir, filename)
            video_paths.append(video_path)
            labels.append(class_id)
    
    return video_paths, labels

def collate_fn(batch):
    # Filter out None values (failed video processing)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    pixel_values = torch.cat([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def main():
    # Free up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print training configuration
    print(f"Training with NUM_FRAMES={NUM_FRAMES}, BATCH_SIZE={BATCH_SIZE}")
    if USE_MIDDLE_50_PERCENT:
        print("Using only the middle 50% of frames from each video")
    
    # Collect video paths and labels
    video_paths, labels = collect_videos()
    print(f"Found {len(video_paths)} videos")
    
    # Split into train and validation sets
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = BadmintonVideoDataset(train_videos, train_labels, processor)
    val_dataset = BadmintonVideoDataset(val_videos, val_labels, processor)
    
    # Load model
    model = AutoModelForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # Needed when changing the number of classes
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        # Memory optimization settings
        fp16=False,  # Disable mixed precision for MPS
        use_mps_device=torch.backends.mps.is_available(),  # Use MPS if available
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch",  # Use memory-efficient optimizer
        dataloader_num_workers=0,  # Don't use multiple workers
        report_to="none",  # Disable reporting to save memory
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")

if __name__ == "__main__":
    main() 