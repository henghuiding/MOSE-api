import os
from PIL import Image
import glob

def reshape_gifs_to_1920x1080_ratio():
    """
    Reshape all GIF images in assets/mosev2 to maintain the same aspect ratio as 1920x1080
    """
    # Target aspect ratio (1920x1080 = 16:9)
    target_ratio = 1920 / 1080
    
    # Path to the assets/mosev2 directory
    assets_dir = "assets/mosev2"
    
    if not os.path.exists(assets_dir):
        print(f"Directory {assets_dir} does not exist!")
        return
    
    # Find all GIF files in the directory
    gif_files = glob.glob(os.path.join(assets_dir, "*.gif"))
    
    if not gif_files:
        print(f"No GIF files found in {assets_dir}")
        return
    
    print(f"Found {len(gif_files)} GIF files to process...")
    
    for gif_path in gif_files:
        try:
            # Open the GIF
            with Image.open(gif_path) as img:
                # Get original dimensions
                original_width, original_height = img.size
                original_ratio = original_width / original_height
                
                print(f"Processing {os.path.basename(gif_path)}: {original_width}x{original_height}")
                
                # Calculate new dimensions maintaining 16:9 ratio
                if original_ratio > target_ratio:
                    # Image is wider than target ratio, fit by height
                    new_height = original_height
                    new_width = int(new_height * target_ratio)
                else:
                    # Image is taller than target ratio, fit by width
                    new_width = original_width
                    new_height = int(new_width / target_ratio)
                
                # Create a list to store all frames
                frames = []
                durations = []
                
                # Process each frame in the GIF
                try:
                    while True:
                        # Get frame duration
                        duration = img.info.get('duration', 100)
                        durations.append(duration)
                        
                        # Resize frame to new dimensions
                        resized_frame = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        frames.append(resized_frame.copy())
                        
                        # Move to next frame
                        img.seek(img.tell() + 1)
                        
                except EOFError:
                    # End of frames
                    pass
                
                # Save the resized GIF with loop enabled
                if frames:
                    output_path = gif_path  # Overwrite original file
                    frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=0,  # 0 means infinite loop
                        optimize=True
                    )
                    print(f"Resized {os.path.basename(gif_path)} to {new_width}x{new_height} (looping enabled)")
                
        except Exception as e:
            print(f"Error processing {gif_path}: {str(e)}")
    
    print("Finished processing all GIF files!")

if __name__ == "__main__":
    reshape_gifs_to_1920x1080_ratio()

