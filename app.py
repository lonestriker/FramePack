from flask import Flask, render_template, request, url_for, send_file, jsonify, redirect, flash
import os
import threading
import queue
import subprocess
import uuid
import time
import json
import math
import logging
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages

# --- Logging Setup --- 
logger = logging.getLogger('framepack')
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Log file setup will happen after config load

# --- Configuration Loading ---
def load_config():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            if not isinstance(config.get('API_SERVERS'), list):
                raise ValueError("'API_SERVERS' key missing or not a list in config.json")
            # Add default for logging flag if missing
            if 'ENABLE_JOB_LOGGING' not in config:
                 config['ENABLE_JOB_LOGGING'] = True
            return config
    except FileNotFoundError:
        print("ERROR: config.json not found. Please create it based on config.json.example.")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Could not parse config.json: {e}")
        exit(1)

config_data = load_config()
API_SERVERS = config_data.get('API_SERVERS', []) # Get servers from loaded config
ENABLE_JOB_LOGGING = config_data.get('ENABLE_JOB_LOGGING', True) # Get logging flag

# --- Options Loading/Saving ---
OPTIONS_FILE = 'options.json'
DEFAULT_OPTIONS = {
    'prompt': '',
    'duration': 5,
    'use_teacache': True # Corresponds to NOT using --no-teacache
}

def load_options():
    try:
        if os.path.exists(OPTIONS_FILE):
            with open(OPTIONS_FILE, 'r') as f:
                loaded = json.load(f)
                # Validate and merge with defaults
                options = DEFAULT_OPTIONS.copy()
                options.update({k: loaded[k] for k in DEFAULT_OPTIONS if k in loaded})
                # Ensure duration is within bounds
                options['duration'] = max(1, min(MAX_DURATION, int(options.get('duration', DEFAULT_DURATION))))
                # Ensure teacache is boolean
                options['use_teacache'] = bool(options.get('use_teacache', True))
                return options
        else:
            return DEFAULT_OPTIONS.copy()
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Warning: Could not load or parse {OPTIONS_FILE}, using defaults. Error: {e}")
        return DEFAULT_OPTIONS.copy()

def save_options(options):
    try:
        with open(OPTIONS_FILE, 'w') as f:
            json.dump(options, f, indent=4)
    except IOError as e:
        print(f"Error: Could not save options to {OPTIONS_FILE}: {e}")
        if ENABLE_JOB_LOGGING:
            logger.error(f"Failed to save options to {OPTIONS_FILE}: {e}")

# Load initial options
app_options = load_options()

# --- Finalize Logging Setup (after config load) ---
if ENABLE_JOB_LOGGING:
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_filename = f"logs/framepack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info("--- Application Started --- Log file created.")
else:
    logger.addHandler(logging.NullHandler()) # Prevent 'no handlers' warning if disabled

# Define retry constants
BASE_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 60 # seconds

# Configuration Constants (can still be defined here if needed)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
MAX_DURATION = 120
DEFAULT_DURATION = 5

# --- In-memory state (for simplicity, replace with DB/proper state management later) ---
job_queue = queue.Queue()
# Updated job_status structure
job_status = {} # {job_id: {'status': 'queued'/'running'/'completed'/'failed_will_retry', 'output': [], 'params': {}}}
# Replace active_servers with server_status
server_status = {
    server: {'available': True, 'fail_count': 0, 'next_retry_time': 0}
    for server in API_SERVERS
}
server_lock = threading.Lock() # Keep the lock

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        directory = request.form.get('directory')
        if directory and os.path.isdir(directory):
            return redirect(url_for('view_dir', directory=directory)) # Correctly use redirect()
        else:
            # Handle invalid directory path
            return render_template('index.html', error="Invalid directory path")
    return render_template('index.html')

@app.route('/view_dir')
def view_dir():
    directory = request.args.get('directory')
    if not directory or not os.path.isdir(directory):
        # Redirecting to index if directory is invalid
        return redirect(url_for('index'))

    images = []
    error_message = None
    try:
        for fname in os.listdir(directory):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                 images.append({'name': fname, 'path': os.path.join(directory, fname)})
    except OSError as e:
        error_message = f"Error accessing directory: {e}" # Store error message

    # Pass durations to the template for the modal
    return render_template('view_dir.html', 
                           directory=directory, 
                           images=images, 
                           error=error_message, 
                           max_duration=MAX_DURATION, 
                           default_duration=app_options['duration'], # Use loaded default duration
                           default_prompt=app_options['prompt']) # Pass default prompt

@app.route('/options', methods=['GET', 'POST'])
def options_page():
    global app_options
    if request.method == 'POST':
        try:
            new_prompt = request.form.get('prompt', '')
            new_duration = request.form.get('duration', DEFAULT_OPTIONS['duration'], type=int)
            # Checkbox value is 'on' if checked, None otherwise
            new_use_teacache = request.form.get('use_teacache') == 'on' 

            # Validate duration
            if not 1 <= new_duration <= MAX_DURATION:
                flash(f'Invalid duration. Must be between 1 and {MAX_DURATION}. Using default.', 'warning')
                new_duration = DEFAULT_OPTIONS['duration']

            app_options['prompt'] = new_prompt
            app_options['duration'] = new_duration
            app_options['use_teacache'] = new_use_teacache
            
            save_options(app_options)
            flash('Options saved successfully!', 'success')
            if ENABLE_JOB_LOGGING:
                logger.info(f"Options updated: Prompt='{new_prompt}', Duration={new_duration}, UseTeaCache={new_use_teacache}")

        except Exception as e:
             flash(f'Error saving options: {e}', 'error')
             if ENABLE_JOB_LOGGING:
                 logger.error(f"Error saving options via web UI: {e}")

        return redirect(url_for('options_page')) # Redirect to refresh page after POST

    # GET request
    return render_template('options.html', options=app_options, max_duration=MAX_DURATION)

@app.route('/generate', methods=['POST'])
def generate():
    global app_options
    image_path = request.form.get('image_path')
    # Use submitted prompt, fallback to saved default, fallback to empty string
    prompt = request.form.get('prompt', app_options['prompt']).strip() 
    # Use submitted duration, fallback to saved default
    duration = request.form.get('duration', app_options['duration'], type=int) 
    # Use the globally configured teacache setting for this job
    use_teacache_for_job = app_options['use_teacache']

    if not image_path or not os.path.isfile(image_path):
        return jsonify({'error': 'Invalid image path'}), 400

    # Validate duration
    if not 1 <= duration <= MAX_DURATION:
        return jsonify({'error': f'Invalid duration. Must be between 1 and {MAX_DURATION}'}), 400

    job_id = str(uuid.uuid4())
    params = {
        'image_path': image_path,
        'prompt': prompt,
        'duration': duration,
        'use_teacache': use_teacache_for_job # Store the effective setting for this job
    }
    job_status[job_id] = {'status': 'queued', 'output': [], 'params': params}
    job_queue.put((job_id, params))

    if ENABLE_JOB_LOGGING:
        filename = os.path.basename(image_path)
        # Log the effective parameters used for the job
        logger.info(f"Job Submitted - ID: {job_id}, Image: {filename}, Prompt: '{params['prompt']}', Duration: {params['duration']}s, UseTeaCache: {params['use_teacache']}")

    return jsonify({'job_id': job_id, 'status': 'queued'}), 202 # 202 Accepted

# --- New API Route for Job Status ---
@app.route('/api/jobs')
def api_jobs():
    # Return a copy to avoid issues if dict changes during serialization
    return jsonify(dict(job_status))

# --- New Route for Server Status ---
@app.route('/api/server_status')
def api_server_status():
    with server_lock:
        # Return a copy to ensure thread safety during serialization
        # Format timestamps for better readability if desired
        status_copy = {}
        current_time = time.time()
        for server, status in server_status.items():
            status_copy[server] = status.copy() # Create a copy of the inner dict
            status_copy[server]['next_retry_available_in_seconds'] = max(0, status['next_retry_time'] - current_time)
            status_copy[server]['next_retry_time_readable'] = time.ctime(status['next_retry_time']) if status['next_retry_time'] > 0 else "N/A"
        return jsonify(status_copy)

# --- Route to serve images safely ---
@app.route('/images/<path:filename>')
def serve_image(filename):
    # IMPORTANT: Add security checks here in a real application 
    # to prevent access outside allowed directories.
    # For now, it trusts the path given implicitly.
    # Consider checking if the requested path starts with an allowed base path.
    try:
        return send_file(filename)
    except FileNotFoundError:
        return "Image not found", 404

# --- Background Worker ---

def worker():
    while True:
        job_id, params = job_queue.get() # Blocks until an item is available

        # --- Server Selection ---
        assigned_server = None
        while assigned_server is None:
            current_time = time.time()
            with server_lock:
                # Find the best available server (least recently failed or ready for retry)
                candidates = []
                for server, status in server_status.items():
                    if status['available'] and current_time >= status['next_retry_time']:
                         candidates.append((server, status['fail_count'], status['next_retry_time']))

                if candidates:
                     # Prioritize servers with lower fail counts, then earlier retry times
                     candidates.sort(key=lambda x: (x[1], x[2]))
                     assigned_server = candidates[0][0]
                     server_status[assigned_server]['available'] = False # Mark as busy temporarily
                     print(f"Job {job_id} assigned to server {assigned_server}")
                     if ENABLE_JOB_LOGGING:
                        # Add prompt to Job Assigned log
                        logger.info(f"Job Assigned - ID: {job_id}, Server: {assigned_server}, Prompt: '{params['prompt']}'")
                else:
                     # Check if any server is potentially available but waiting for backoff
                     waiting_servers = any(current_time < s['next_retry_time'] for s in server_status.values() if s['next_retry_time'] > 0)
                     all_servers_busy = all(not s['available'] for s in server_status.values())
                     # Only print waiting message if there are servers in backoff or all busy
                     #if waiting_servers or all_servers_busy:
                          # print(f"No servers immediately available for job {job_id}. Waiting...") # Less verbose debugging
                          # pass
                     time.sleep(0.5) # Wait before checking again


        # --- Job Execution ---
        job_status[job_id]['status'] = 'running'
        # Simplified log message, removed retry count
        # Output now includes assignment info, so logger info added above
        # job_status[job_id]['output'].append(f"Attempting job on server: {assigned_server}")

        job_failed = False
        error_message = ""

        try:
            command = [
                'python',
                'api-client.py',
                '--api_url', assigned_server,
                '--prompt', params['prompt'],
                '--length', str(params['duration']),
                '--image', params['image_path']
            ]
            if not params.get('use_teacache', True): # Default to True if missing
                command.append('--no-teacache')

            # Log the actual command being run
            print(f"Executing command for job {job_id}: {' '.join(command)}")
            if ENABLE_JOB_LOGGING:
                logger.info(f"Executing Command - ID: {job_id}, Server: {assigned_server}, Command: {' '.join(command)}")

            process = subprocess.Popen(command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       text=True,
                                       bufsize=1,
                                       universal_newlines=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__)))

            output_lines = []
            for line in process.stdout:
                stripped_line = line.strip()
                # Limit output stored per job to avoid memory issues? Maybe later.
                if len(job_status[job_id]['output']) < 100: # Limit to 100 lines for now
                    job_status[job_id]['output'].append(stripped_line)
                output_lines.append(stripped_line) # Keep local copy for logging

            process.wait()

            if process.returncode != 0:
                job_failed = True
                # More specific error message
                error_message = f"Job script failed on {assigned_server} with return code: {process.returncode}"
                job_status[job_id]['output'].append(error_message)
            else:
                # Success!
                job_status[job_id]['status'] = 'completed'
                job_status[job_id]['output'].append("Job completed successfully.")
                 # Reset server status on success
                with server_lock:
                    server_status[assigned_server]['fail_count'] = 0
                    server_status[assigned_server]['next_retry_time'] = 0
                    # Ensure available is True after success
                    if not server_status[assigned_server]['available']:
                         server_status[assigned_server]['available'] = True # Should already be false, but make sure
                         print(f"Server {assigned_server} marked available after job success.")
                print(f"Job {job_id} completed successfully on {assigned_server}. Resetting server status.")
                if ENABLE_JOB_LOGGING:
                    # Extract output filename based on api-client.py's specific output format
                    output_filename = "Unknown"
                    output_prefix = "Video file: "
                    for line in output_lines:
                        stripped_line = line.strip()
                        if stripped_line.startswith(output_prefix):
                            full_path = stripped_line[len(output_prefix):].strip()
                            if full_path: # Ensure we got a path
                                output_filename = os.path.basename(full_path)
                                break # Found it

                    # Store the found (or default 'Unknown') filename
                    job_status[job_id]['output_filename'] = output_filename
                    # Add prompt to Job Completed log
                    teacache_status = params.get('use_teacache', True)
                    logger.info(f"Job Completed - ID: {job_id}, Output: {output_filename}, Prompt: '{params['prompt']}', UseTeaCache: {teacache_status}")

        except Exception as e:
            job_failed = True
            error_message = f"Exception during job execution on {assigned_server}: {e}"
            job_status[job_id]['output'].append(error_message)
            print(f"Job {job_id} encountered exception on {assigned_server}: {e}")

        finally:
            server_needs_release = True # Assume server needs release unless failed
            if job_failed:
                # Failure: Penalize server, re-queue job, mark server available for future (after backoff)
                job_status[job_id]['status'] = 'failed_will_retry' # Status indicates it will be tried again
                job_status[job_id]['output'].append(f"Job failed on {assigned_server}. Re-queuing.")
                print(f"Job {job_id} failed on {assigned_server}. Applying backoff and re-queuing.")
                if ENABLE_JOB_LOGGING:
                     logger.warning(f"Job Failed - ID: {job_id}, Server: {assigned_server}. Reason: {error_message}. Applying backoff and re-queuing.")

                # Apply backoff to the server and mark available (respecting backoff time)
                with server_lock:
                    server_status[assigned_server]['fail_count'] += 1
                    # Calculate delay: base * 2^(fails-1), ensure fail_count >= 1
                    delay = min(BASE_RETRY_DELAY * (2 ** max(0, server_status[assigned_server]['fail_count'] - 1)), MAX_RETRY_DELAY)
                    server_status[assigned_server]['next_retry_time'] = time.time() + delay
                    server_status[assigned_server]['available'] = True # It's available, but the selection logic checks next_retry_time
                    print(f"Server {assigned_server} failed. Backoff: {delay:.2f}s. Fail count: {server_status[assigned_server]['fail_count']}. Next available check after: {time.ctime(server_status[assigned_server]['next_retry_time'])} ")
                    if ENABLE_JOB_LOGGING:
                        logger.info(f"Server Backoff - Server: {assigned_server}, FailCount: {server_status[assigned_server]['fail_count']}, Delay: {delay:.2f}s, Next Check: {time.ctime(server_status[assigned_server]['next_retry_time'])}")

                # Re-queue the job IMMEDIATELY (it will wait for an available server)
                job_queue.put((job_id, params))
                if ENABLE_JOB_LOGGING:
                    logger.info(f"Job Re-queued - ID: {job_id}")
                server_needs_release = False # Availability already set in backoff logic

            # Release the server lock only if the job succeeded
            if server_needs_release:
                 with server_lock:
                      # Double-check it should be available
                      if not server_status[assigned_server]['available']:
                         server_status[assigned_server]['available'] = True
                         # Resetting time just in case, although it should have been done on success path
                         server_status[assigned_server]['next_retry_time'] = 0
                         print(f"Server {assigned_server} explicitly released (job success).")


            job_queue.task_done() # Signal task completion for queue management

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    # Start background worker thread(s)
    # Match the number of workers to the number of API servers for concurrency
    num_workers = len(API_SERVERS)
    print(f"Starting {num_workers} worker threads...")
    for i in range(num_workers):
        thread = threading.Thread(target=worker, daemon=True, name=f"Worker-{i+1}")
        thread.start()

    if ENABLE_JOB_LOGGING:
        logger.info(f"Starting Flask server with {num_workers} workers.")
    else:
        print("Job logging disabled via config.")

    if ENABLE_JOB_LOGGING:
        logger.info(f"Loaded options: {app_options}")
    else:
        print(f"Loaded options: {app_options}")
        
    app.run(debug=True, host='0.0.0.0', port=5001) # Run on all available interfaces
