import os
import subprocess
import argparse

def convert_mkv_to_mp4(input_file, output_file=None):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")

    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = base + ".mp4"

    command = [
        "ffmpeg", "-i", input_file,
        "-c", "copy",  # copia audio/video senza ricodifica
        output_file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[OK] Converted: {input_file} → {output_file}")
    except subprocess.CalledProcessError:
        print(f"[ERROR] Conversion failed for {input_file}")

def process_path(path):
    if os.path.isfile(path):
        if path.lower().endswith(".mkv"):
            convert_mkv_to_mp4(path)
        else:
            print(f"[SKIP] {path} is not an MKV file")
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith(".mkv"):
                full_path = os.path.join(path, filename)
                convert_mkv_to_mp4(full_path)
    else:
        print(f"[ERROR] {path} is not a valid file or directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MKV files to MP4 using ffmpeg")
    parser.add_argument("path", type=str, help="Path to an MKV file or a directory containing MKV files")
    args = parser.parse_args()

    process_path(args.path)

