import os
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

def process_videos():
    # Define directories using absolute paths based on the workspace
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "videos", "input")  # Input videos directory, ./videos/input
    output_dir = os.path.join(base_dir, "videos", "output")  # Output videos directory, ./videos/output
    
    # Ensure output directory exists (though the tree shows it exists)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the YOLO model
    model_name = "best.pt"
    print(f"🚀 Loading YOLO model: {model_name}...")
    
    try:
        # ultralytics YOLO() automatically downloads the model if it's a standard one 
        # (like yolov8n.pt) and it's not found locally.
        model = YOLO(model_name)
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure you have internet for first-time model download")
        return
    
    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"⚠️ No video files found in {input_dir}")
        return

    print(f"📂 Found {len(video_files)} videos to process.")
    
    for video_name in video_files:
        input_path = os.path.join(input_dir, video_name)
        
        # Generate timestamped output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{timestamp}_{video_name}"
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\n🎬 Processing: {video_name}")
        print(f"➡️ Saving to: {output_name}")
        
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_name}")
            continue
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use MP4V codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Run inference
            # We use stream=True for memory efficiency when processing many frames
            results = model(frame, verbose=False)
            
            # Annotate frame
            # results[0].plot() handles name and confidence display
            annotated_frame = results[0].plot()
            
            # Write the frame
            out.write(annotated_frame)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                print(f"   Progress: {frame_count}/{total_frames} frames ({fps_processing:.1f} FPS)", end='\r')
        
        # Cleanup for this video
        cap.release()
        out.release()
        
        duration = time.time() - start_time
        print(f"\n✅ Finished {video_name} in {duration:.1f} seconds.")

    print("\n🎉 All videos processed successfully!")

if __name__ == "__main__":
    process_videos()
