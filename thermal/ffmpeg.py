import os

# ffmpegを使って画像を動画に変換
output_dir = 'thermal_output'
os.system(f'ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p thermal_simulation.mp4')