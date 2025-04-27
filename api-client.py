#!/usr/bin/env python
import os
import sys
import time
import argparse
import requests
from pathlib import Path
import json
from tqdm import tqdm
import base64


def parse_args():
    parser = argparse.ArgumentParser(description="FramePack API Client")
    
    # Group for generation parameters
    gen_group = parser.add_argument_group('Generation Options')
    gen_group.add_argument("--api_url", type=str, default="http://127.0.0.1:8001", help="API URL")
    gen_group.add_argument("--url", type=str, required=False, help="URL to image")
    gen_group.add_argument("--image", type=str, required=False, help="Path to input image")
    gen_group.add_argument("--prompt", type=str, help="Text prompt for generation")
    gen_group.add_argument("--seed", type=int, default=31337, help="Random seed")
    gen_group.add_argument("--length", type=float, default=5.0, help="Video length in seconds")
    gen_group.add_argument("--crf", type=int, default=16, help="MP4 compression (lower = better quality)")
    gen_group.add_argument("--gpu-memory", type=float, default=6.0, help="GPU memory preservation in GB")
    gen_group.add_argument("--output-dir", type=str, default="./downloads", help="Directory to save the result")
    gen_group.add_argument("--poll-interval", type=float, default=5.0, help="Status polling interval in seconds")
    gen_group.add_argument("--sync", action="store_true", help="Use synchronous API endpoint (/generate-wait)")
    gen_group.add_argument("--no-teacache", action="store_true", default=False,
                        help="Disable TeaCache (slower generation but may improve hand/finger quality)")

    # Group for cancellation
    cancel_group = parser.add_argument_group('Cancellation Options')
    cancel_group.add_argument("--cancel", type=str, metavar="JOB_ID", help="Cancel a running job by its ID")
    cancel_group.add_argument("--cancel-current", action="store_true", help="Cancel the currently active job on the server")

    # Group for listing jobs
    list_group = parser.add_argument_group('Listing Option')
    list_group.add_argument("--list-jobs", action="store_true", help="List all jobs on the server")


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


def submit_generation_job_sync(api_url, image_url, image_path, prompt, seed, length, crf, gpu_memory, use_teacache):
    """Submit a generation job to the synchronous API endpoint and return the video directly"""
    try:
        data = {
            'prompt': prompt,
            'seed': seed,
            'total_second_length': length,
            'mp4_crf': crf,
            'gpu_memory_preservation': gpu_memory,
            'use_teacache': use_teacache
        }
        
        print(f"Submitting job to {api_url}/generate-wait (synchronous mode)")
        print(f"Prompt: {prompt}")
        print(f"Seed: {seed}, Length: {length}s, CRF: {crf}")
        print(f"TeaCache: {'enabled' if use_teacache else 'disabled'}")
        print("Waiting for generation to complete. This may take several minutes...")
        
        # Progress indicator for the wait
        spinner = ['|', '/', '-', '\\']
        i = 0
        
        # Either submit a file or a URL with a long timeout
        if image_path:
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                
                # Start a session with a very long timeout for the sync request
                with requests.Session() as session:
                    session.timeout = 3600  # 1 hour timeout
                    
                    # Create a stream to monitor progress
                    with session.post(f"{api_url}/generate-wait", 
                                     files=files, 
                                     data=data, 
                                     stream=True) as response:
                        if response.status_code != 200:
                            print(f"Error: API returned status code {response.status_code}")
                            print(response.text)
                            sys.exit(1)
                        
                        # Collect the complete response
                        content = b''
                        print("Processing... ", end='', flush=True)
                        for chunk in response.iter_content(chunk_size=8192):
                            # Show a spinner to indicate it's still working
                            print(f"\rProcessing... {spinner[i % 4]}", end='', flush=True)
                            i += 1
                            content += chunk
                        print("\rProcessing... Done!       ")
                        
                        # Parse the JSON response
                        try:
                            result = json.loads(content)
                            return result
                        except json.JSONDecodeError:
                            print("Error: Received invalid JSON response")
                            sys.exit(1)
                        
        elif image_url:
            data['url'] = image_url
            
            # Start a session with a very long timeout for the sync request
            with requests.Session() as session:
                session.timeout = 3600  # 1 hour timeout
                
                # Create a stream to monitor progress
                with session.post(f"{api_url}/generate-wait", 
                                 data=data, 
                                 stream=True) as response:
                    if response.status_code != 200:
                        print(f"Error: API returned status code {response.status_code}")
                        print(response.text)
                        sys.exit(1)
                    
                    # Collect the complete response
                    content = b''
                    print("Processing... ", end='', flush=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        # Show a spinner to indicate it's still working
                        print(f"\rProcessing... {spinner[i % 4]}", end='', flush=True)
                        i += 1
                        content += chunk
                    print("\rProcessing... Done!       ")
                    
                    # Parse the JSON response
                    try:
                        result = json.loads(content)
                        return result
                    except json.JSONDecodeError:
                        print("Error: Received invalid JSON response")
                        sys.exit(1)
        else:
            print("Error: Neither image path nor URL provided")
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"Error submitting job: {str(e)}")
        sys.exit(1)


def submit_generation_job(api_url, image_url, image_path, prompt, seed, length, crf, gpu_memory, use_teacache):
    """Submit a generation job to the API and return the job ID"""
    try:
        data = {
            'prompt': prompt,
            'seed': seed,
            'total_second_length': length,
            'mp4_crf': crf,
            'gpu_memory_preservation': gpu_memory,
            'use_teacache': use_teacache
        }
        
        print(f"Submitting job to {api_url}/generate")
        print(f"Prompt: {prompt}")
        print(f"Seed: {seed}, Length: {length}s, CRF: {crf}")
        print(f"TeaCache: {'enabled' if use_teacache else 'disabled'}")
        
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


def cancel_job(api_url, job_id):
    """Send a cancellation request to the API"""
    try:
        print(f"Sending cancellation request for job ID: {job_id} to {api_url}/cancel/{job_id}")
        response = requests.post(f"{api_url}/cancel/{job_id}", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Cancellation request sent successfully. Server response: Status={result.get('status', 'N/A')}, Message='{result.get('message', 'N/A')}'")
            return True
        elif response.status_code == 404:
            print(f"Error: Job ID '{job_id}' not found on the server.")
            return False
        elif response.status_code == 400:
             print(f"Error: Job '{job_id}' could not be cancelled (likely already completed, failed, or cancelled).")
             print(response.text)
             return False
        else:
            print(f"Error: API returned status code {response.status_code} during cancellation.")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error sending cancellation request: {str(e)}")
        return False


def cancel_current_job(api_url):
    """Send a request to cancel the currently active job"""
    try:
        print(f"Sending request to cancel current active job to {api_url}/cancel-current")
        response = requests.post(f"{api_url}/cancel-current", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Cancellation request sent successfully. Server response for job '{result.get('job_id', 'N/A')}': Status={result.get('status', 'N/A')}, Message='{result.get('message', 'N/A')}'")
            return True
        elif response.status_code == 404:
            print(f"Error: Server reported no currently active job.")
            return False
        elif response.status_code == 400:
             print(f"Error: Current job could not be cancelled (likely already completed, failed, or cancelled).")
             print(response.text)
             return False
        else:
            print(f"Error: API returned status code {response.status_code} during current job cancellation.")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error sending cancel-current request: {str(e)}")
        return False


def poll_job_status(api_url, job_id, poll_interval):
    """Poll the job status until completion or failure, printing progress."""
    print(f"Polling job status every {poll_interval} seconds... Press Ctrl+C to attempt cancellation.")
    
    # Use simple print instead of tqdm for subprocess compatibility
    last_progress = -1
    last_message = ""
    
    session = requests.Session()
    
    while True:
        try:
            response = session.get(f"{api_url}/status/{job_id}", timeout=30)
            
            if response.status_code != 200:
                # Print error to stderr? Or just continue? Print to stdout for now.
                print(f"Error getting status: API returned status code {response.status_code}")
                print(response.text)
                time.sleep(poll_interval)
                continue
                
            status_data = response.json()
            current_progress = status_data.get('progress', 0)
            current_message = status_data.get('message', '')
            current_status = status_data.get('status', 'unknown')

            # --- Print Progress Update Line ---
            # Only print if progress or message changed to avoid spamming stdout
            if current_progress != last_progress or current_message != last_message:
                 print(f"PROGRESS::{current_progress}::{current_message}", flush=True)
                 last_progress = current_progress
                 last_message = current_message
            # --- End Progress Update Line ---

            # Check if processing is done
            if current_status == 'completed':
                print(f"Job completed. Final message: {current_message}", flush=True)
                return status_data.get('video_url') # Return URL on success
            elif current_status in ['failed', 'cancelled']: 
                print(f"Job finished with status: {current_status}. Message: {current_message}", flush=True)
                # If cancelled, we don't treat it as a script failure
                if current_status == 'failed':
                     sys.exit(1) # Exit with error code for failure
                else:
                     return None # Indicate cancellation

        except requests.exceptions.RequestException as e:
            print(f"Error polling status (will retry): {str(e)}", flush=True)
        
        # Wait interval, but allow interruption for cancellation
        try:
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received during polling.", flush=True)
            try:
                # Don't prompt in subprocess mode, just attempt cancel
                print(f"Attempting to cancel job {job_id}...", flush=True)
                if cancel_job(api_url, job_id):
                    print("Cancellation requested. Exiting polling.", flush=True)
                    return None # Indicate cancellation attempt
                else:
                    print("Failed to send cancellation request. Continuing polling...", flush=True)
            except EOFError: 
                 print("\nInput stream closed, cannot confirm cancellation. Continuing polling...", flush=True)


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


def save_base64_video(base64_data, output_dir, seed, prompt):
    """Save a base64-encoded video to file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Generate a filename based on seed and timestamp
        timestamp = int(time.time())
        filename = f"video_{seed}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        print(f"Saving video to {output_path}...")
        
        # Decode the base64 data
        video_bytes = base64.b64decode(base64_data)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(video_bytes)
        
        print(f"Video saved successfully to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving video: {str(e)}")
        sys.exit(1)


def list_jobs(api_url):
    """Fetch and display the list of jobs from the API"""
    try:
        print(f"Fetching job list from {api_url}/jobs")
        response = requests.get(f"{api_url}/jobs", timeout=30)
        
        if response.status_code == 200:
            jobs_list = response.json()
            if not jobs_list:
                print("No jobs found on the server.")
                return True
                
            print("\n--- Job List ---")
            # Sort client-side just in case server doesn't guarantee order
            jobs_list.sort(key=lambda x: x.get("creation_time", ""), reverse=True) 
            for job in jobs_list:
                print(f"  ID: {job.get('job_id', 'N/A')}")
                print(f"    Status: {job.get('status', 'N/A')}")
                print(f"    Progress: {job.get('progress', 0)}%")
                print(f"    Created: {job.get('creation_time', 'N/A')}")
                print(f"    Message: {job.get('message', '')}")
                if job.get('video_url'):
                    print(f"    Video URL: {job.get('video_url')}")
                print("-" * 16)
            return True
        else:
            print(f"Error: API returned status code {response.status_code} when listing jobs.")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching job list: {str(e)}")
        return False


def main():
    args = parse_args()

    # Handle mutually exclusive options
    action_count = sum([
        args.cancel is not None, 
        args.list_jobs, 
        args.cancel_current,
        (args.prompt is not None or args.image is not None or args.url is not None) # Generation options grouped
    ])
    if action_count > 1:
        print("Error: --cancel, --list-jobs, --cancel-current, and generation options (like --prompt) are mutually exclusive.")
        sys.exit(1)
    if action_count == 0:
         print("Error: No action specified. Use --prompt/--image/--url for generation, --list-jobs, --cancel JOB_ID, or --cancel-current.")
         sys.exit(1)


    # Handle list jobs request
    if args.list_jobs:
        if not args.api_url:
             print("Error: --api_url is required for listing jobs.")
             sys.exit(1)
        if list_jobs(args.api_url):
            sys.exit(0)
        else:
            sys.exit(1)

    # Handle cancel specific job request
    if args.cancel:
        if any([args.image, args.url, args.prompt, args.sync]):
             print("Error: --cancel cannot be used with generation options (--image, --url, --prompt, --sync, etc.).")
             sys.exit(1)
        if not args.api_url:
             print("Error: --api_url is required for cancellation.")
             sys.exit(1)
             
        if cancel_job(args.api_url, args.cancel):
             sys.exit(0) # Success
        else:
             sys.exit(1) # Failure

    # Handle cancel current job request
    if args.cancel_current:
        if not args.api_url:
             print("Error: --api_url is required for cancelling the current job.")
             sys.exit(1)
        if cancel_current_job(args.api_url):
             sys.exit(0) # Success
        else:
             sys.exit(1) # Failure


    # --- Proceed with generation if not cancelling or listing ---
    
    # Convert no-teacache flag to use_teacache boolean
    use_teacache = not args.no_teacache
    
    # Validate generation arguments
    if not args.prompt:
         print("Error: --prompt is required for generation.")
         sys.exit(1)
    if not args.image and not args.url:
        print("Error: Either --image or --url must be provided for generation.")
        sys.exit(1)
    if args.image and args.url:
        print("Error: Cannot provide both --image and --url. Choose one.")
        sys.exit(1)
    
    # Validate image file if provided
    if args.image:
        check_image_file(args.image)
    
    # Handle the sync mode
    if args.sync:
        print("--- Synchronous Mode ---")
        print("Note: Pressing Ctrl+C will only stop this client.")
        print(f"To cancel the server-side job, run separately: python {sys.argv[0]} --api_url {args.api_url} --cancel JOB_ID")
        # Use the synchronous API endpoint
        response_data = submit_generation_job_sync(
            args.api_url,
            args.url,
            args.image, 
            args.prompt, 
            args.seed, 
            args.length,
            args.crf,
            args.gpu_memory,
            use_teacache
        )
        
        # Process the response
        try:
            # Extract video data from the response
            video_base64 = response_data["images"][0]
            duration = response_data["parameters"]["duration"]
            
            # Save the video file
            output_path = save_base64_video(video_base64, args.output_dir, args.seed, args.prompt)
            
            print("\nGeneration completed successfully!")
            print(f"Video duration: {duration} seconds")
            print(f"Video file: {output_path}")
            if output_path:
                print(f"OUTPUT_FILE_PATH::{output_path}") 
        except (KeyError, IndexError) as e:
            print(f"Error parsing API response: {str(e)}")
            print("Response:", json.dumps(response_data, indent=2))
            sys.exit(1)
            
    else:
        # --- Asynchronous Mode (Modified) ---
        print("--- Asynchronous Mode ---")
        job_id = submit_generation_job(
            args.api_url,
            args.url,
            args.image, 
            args.prompt, 
            args.seed, 
            args.length,
            args.crf,
            args.gpu_memory,
            use_teacache
        )
        
        # Poll for status and print progress directly
        video_url = poll_job_status(args.api_url, job_id, args.poll_interval)
        
        # Download the result if polling completed successfully (didn't return None or exit)
        if video_url:
            output_path = download_result(args.api_url, video_url, args.output_dir)
            # Print final success message (already printed by poll_job_status)
            # print("\nGeneration completed successfully!") 
            print(f"Video file: {output_path}", flush=True)
            # --- Print special marker for app.py to find the output path ---
            # Ensure this is the *last* thing printed on success
            if output_path:
                print(f"OUTPUT_FILE_PATH::{output_path}", flush=True) 
            # --- End special marker ---
        elif video_url is None: # Indicates cancellation occurred during polling
             print("\nJob was cancelled.", flush=True)
             sys.exit(0) # Exit gracefully after cancellation
        # If poll_job_status exited due to failure, sys.exit(1) was already called
        # --- End Asynchronous Mode (Modified) ---


if __name__ == "__main__":
    # Initialize last_message attribute to avoid AttributeError
    poll_job_status.last_message = None
    
    try:
        main()
    except KeyboardInterrupt:
        # Catch final KeyboardInterrupt if it happens outside the polling loop's specific handler
        print("\nOperation interrupted by user.")
        sys.exit(0)
    except SystemExit as e:
         # Allow sys.exit calls to propagate naturally
         sys.exit(e.code)
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)
