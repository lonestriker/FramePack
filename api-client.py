#!/usr/bin/env python
import os
import sys
import time
import argparse
import requests
from pathlib import Path
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="FramePack API Client")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8001", help="API URL")
    parser.add_argument("--url", type=str, required=False, help="URL to image")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=31337, help="Random seed")
    parser.add_argument("--length", type=float, default=5.0, help="Video length in seconds")
    parser.add_argument("--crf", type=int, default=16, help="MP4 compression (lower = better quality)")
    parser.add_argument("--gpu-memory", type=float, default=6.0, help="GPU memory preservation in GB")
    parser.add_argument("--output-dir", type=str, default="./downloads", help="Directory to save the result")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Status polling interval in seconds")
    return parser.parse_args()


def check_image_file(image_path):
    """Check if image file exists and has a valid extension"""
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Error: Image file must be one of: {', '.join(valid_extensions)}")
        sys.exit(1)


def submit_generation_job(api_url, image_url, image_path, prompt, seed, length, crf, gpu_memory):
    """Submit a generation job to the API and return the job ID"""
    try:
        data = {
            'prompt': prompt,
            'seed': seed,
            'total_second_length': length,
            'mp4_crf': crf,
            'gpu_memory_preservation': gpu_memory
        }
        
        print(f"Submitting job to {api_url}/generate")
        print(f"Prompt: {prompt}")
        print(f"Seed: {seed}, Length: {length}s, CRF: {crf}")
        
        # Either submit a file or a URL
        if image_path:
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                response = requests.post(f"{api_url}/generate", files=files, data=data, timeout=180)
        elif image_url:
            data['url'] = image_url
            response = requests.post(f"{api_url}/generate", data=data, timeout=180)
        else:
            print("Error: Neither image path nor URL provided")
            sys.exit(1)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            sys.exit(1)
            
        result = response.json()
        print(f"Job submitted successfully. Job ID: {result['job_id']}")
        return result['job_id']
        
    except requests.exceptions.RequestException as e:
        print(f"Error submitting job: {str(e)}")
        sys.exit(1)


def poll_job_status(api_url, job_id, poll_interval):
    """Poll the job status until completion or failure"""
    print(f"Polling job status every {poll_interval} seconds...")
    
    progress_bar = tqdm(total=100, desc="Processing", unit="%")
    last_progress = 0
    
    session = requests.Session()
    
    while True:
        try:
            response = session.get(f"{api_url}/status/{job_id}", timeout=30)
            
            if response.status_code != 200:
                print(f"Error getting status: API returned status code {response.status_code}")
                print(response.text)
                time.sleep(poll_interval)
                continue
                
            status_data = response.json()
            
            # Update progress bar
            if status_data['progress'] > last_progress:
                progress_bar.update(status_data['progress'] - last_progress)
                last_progress = status_data['progress']
            
            # Print message if it's changed
            if hasattr(poll_job_status, 'last_message') and poll_job_status.last_message == status_data['message']:
                pass  # Don't print the same message again
            else:
                poll_job_status.last_message = status_data['message']
                print(f"\nStatus: {status_data['status']} - {status_data['message']}")
            
            # Check if processing is done
            if status_data['status'] == 'completed':
                progress_bar.close()
                return status_data['video_url']
            elif status_data['status'] == 'failed':
                progress_bar.close()
                print(f"Job failed: {status_data['message']}")
                sys.exit(1)
                
        except requests.exceptions.RequestException as e:
            print(f"Error polling status (will retry): {str(e)}")
            
        time.sleep(poll_interval)


def download_result(api_url, video_url, output_dir):
    """Download the generated video"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Extract filename from URL
        filename = os.path.basename(video_url)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Downloading video to {output_path}...")
        
        # Stream the download with progress reporting
        with requests.get(f"{api_url}{video_url}", stream=True, timeout=180) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        
        print(f"Video downloaded successfully to {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading result: {str(e)}")
        sys.exit(1)


def main():
    args = parse_args()
    
    # Validate that either image file or URL is provided
    if not args.image and not args.url:
        print("Error: Either --image or --url must be provided")
        sys.exit(1)
    
    if args.image and args.url:
        print("Error: Cannot provide both --image and --url. Choose one.")
        sys.exit(1)
    
    # Validate image file if provided
    if args.image:
        check_image_file(args.image)
    
    # Submit job
    job_id = submit_generation_job(
        args.api_url,
        args.url,
        args.image, 
        args.prompt, 
        args.seed, 
        args.length,
        args.crf,
        args.gpu_memory
    )
    
    # Poll for status - fix the API URL parameter
    video_url = poll_job_status(args.api_url, job_id, args.poll_interval)
    
    # Download the result - fix the API URL parameter
    if video_url:
        output_path = download_result(args.api_url, video_url, args.output_dir)
        print("\nGeneration completed successfully!")
        print(f"Video file: {output_path}")


if __name__ == "__main__":
    # Initialize last_message attribute to avoid AttributeError
    poll_job_status.last_message = None
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        sys.exit(0)
