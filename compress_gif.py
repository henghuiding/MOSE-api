import os
from PIL import Image
import glob

def compress_gifs():
    """
    Compress all GIF images in assets/mosev2 directory
    """
    # Path to the assets/mosev2 directory
    assets_dir = "/home/knying/yingkaining/project/github_released/MOSE-api/assets/mosev2"
    
    if not os.path.exists(assets_dir):
        print(f"Directory {assets_dir} does not exist!")
        return
    
    # Find all GIF files in the directory
    gif_files = glob.glob(os.path.join(assets_dir, "*.gif"))
    
    if not gif_files:
        print(f"No GIF files found in {assets_dir}")
        return
    
    print(f"Found {len(gif_files)} GIF files to compress...")
    
    for gif_path in gif_files:
        try:
            print(f"Compressing {os.path.basename(gif_path)}...")
            
            # Open the GIF
            with Image.open(gif_path) as img:
                # Get original file size
                original_size = os.path.getsize(gif_path)
                
                # Create a list to store all frames
                frames = []
                durations = []
                
                # Process each frame in the GIF
                try:
                    while True:
                        # Get frame duration
                        duration = img.info.get('duration', 100)
                        durations.append(duration)
                        
                        # Convert frame to RGB if necessary and add to frames
                        frame = img.convert('RGB')
                        frames.append(frame.copy())
                        
                        # Move to next frame
                        img.seek(img.tell() + 1)
                        
                except EOFError:
                    # End of frames
                    pass
                
                # Save the compressed GIF
                if frames:
                    # Convert back to P mode (palette mode) for better compression
                    compressed_frames = []
                    for frame in frames:
                        # Quantize to reduce colors (improves compression)
                        quantized = frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                        compressed_frames.append(quantized)
                    
                    # Save with compression settings
                    compressed_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=compressed_frames[1:],
                        duration=durations,
                        loop=0,  # 0 means infinite loop
                        optimize=True,  # Enable optimization
                        quality=85  # Reduce quality for better compression
                    )
                    
                    # Get new file size
                    new_size = os.path.getsize(gif_path)
                    compression_ratio = (1 - new_size / original_size) * 100
                    
                    print(f"Compressed {os.path.basename(gif_path)}: {original_size} bytes -> {new_size} bytes ({compression_ratio:.1f}% reduction)")
                
        except Exception as e:
            print(f"Error compressing {gif_path}: {str(e)}")
    
    print("Finished compressing all GIF files!")

if __name__ == "__main__":
    compress_gifs()
