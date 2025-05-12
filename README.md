# TimeSformer Fine-tuning for Badminton Shot Classification

This repository contains code to fine-tune the TimeSformer model for classifying different badminton shots (forehand, backhand, etc.) from video data.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Organize your badminton videos in the following structure:
   ```
   badminton_dataset/
   ├── forehand_lift/
   │   ├── video1.mp4
   │   ├── video2.mp4
   │   └── ...
   ├── backhand_drive/
   │   ├── video1.mp4
   │   ├── video2.mp4
   │   └── ...
   └── ...
   ```

## Usage

### Running the Python Script

```
python finetune_timesformer.py
```

### Converting to a Notebook

To convert the Python script to a Jupyter notebook:

1. Install nbconvert if you don't have it:
   ```
   pip install nbconvert
   ```

2. Convert the script to a notebook:
   ```
   jupyter nbconvert --to notebook --execute finetune_timesformer.py
   ```

   Or simply create a new notebook and copy the code sections.

## Parameters

You can adjust the following parameters in the script:
- `NUM_FRAMES`: Number of frames to sample from each video
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate for training
- `NUM_EPOCHS`: Number of training epochs

## Output

The fine-tuned model will be saved in the `finetuned_timesformer/final_model` directory. 