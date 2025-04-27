from flask import Flask, render_template, request, url_for, send_file, jsonify, redirect, flash, session
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
import argparse # Import argparse

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages and session

# --- Logging Setup --- 
logger = logging.getLogger('framepack')
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Log file setup will happen after config load

# --- Configuration Loading ---
CONFIG_FILE = 'config.json'
CONFIG_EXAMPLE_FILE = 'config.json.example'
DEFAULT_CONFIG = {
    'API_SERVERS': ['http://127.0.0.1:8001'], # Default if no files exist
    'ENABLE_JOB_LOGGING': True
}

def save_config(config_data_to_save):
    """Saves the configuration data to config.json"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data_to_save, f, indent=4)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    except IOError as e:
        print(f"ERROR: Could not save config to {CONFIG_FILE}: {e}")
        logger.error(f"Failed to save config to {CONFIG_FILE}: {e}")

def load_config():
    """Loads configuration, falling back to example or defaults."""
    config_to_load = None
    loaded_from = ""

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_to_load = json.load(f)
                loaded_from = CONFIG_FILE
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load or parse {CONFIG_FILE}, trying example. Error: {e}")
            logger.warning(f"Could not load or parse {CONFIG_FILE}, trying example. Error: {e}")

    if config_to_load is None and os.path.exists(CONFIG_EXAMPLE_FILE):
        try:
            with open(CONFIG_EXAMPLE_FILE, 'r') as f:
                config_to_load = json.load(f)
                loaded_from = CONFIG_EXAMPLE_FILE
                print(f"Info: Loaded configuration from {CONFIG_EXAMPLE_FILE}. Saving as {CONFIG_FILE}.")
                logger.info(f"Loaded configuration from {CONFIG_EXAMPLE_FILE}. Saving as {CONFIG_FILE}.")
                # Save the example content as the actual config file
                save_config(config_to_load)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load or parse {CONFIG_EXAMPLE_FILE}, using defaults. Error: {e}")
            logger.warning(f"Could not load or parse {CONFIG_EXAMPLE_FILE}, using defaults. Error: {e}")

    if config_to_load is None:
        print(f"Warning: No config files found. Using default configuration and creating {CONFIG_FILE}.")
        logger.warning(f"No config files found. Using default configuration and creating {CONFIG_FILE}.")
        config_to_load = DEFAULT_CONFIG.copy()
        loaded_from = "Defaults"
        save_config(config_to_load) # Create the file with defaults

    # Validate and apply defaults for missing keys
    if not isinstance(config_to_load.get('API_SERVERS'), list):
        print(f"Warning: 'API_SERVERS' missing or invalid in {loaded_from}. Using default.")
        logger.warning(f"'API_SERVERS' missing or invalid in {loaded_from}. Using default.")
        config_to_load['API_SERVERS'] = DEFAULT_CONFIG['API_SERVERS']
        
    if 'ENABLE_JOB_LOGGING' not in config_to_load or not isinstance(config_to_load.get('ENABLE_JOB_LOGGING'), bool):
         print(f"Warning: 'ENABLE_JOB_LOGGING' missing or invalid in {loaded_from}. Using default ({DEFAULT_CONFIG['ENABLE_JOB_LOGGING']}).")
         logger.warning(f"'ENABLE_JOB_LOGGING' missing or invalid in {loaded_from}. Using default ({DEFAULT_CONFIG['ENABLE_JOB_LOGGING']}).")
         config_to_load['ENABLE_JOB_LOGGING'] = DEFAULT_CONFIG['ENABLE_JOB_LOGGING']

    print(f"Configuration loaded from: {loaded_from}")
    logger.info(f"Configuration loaded from: {loaded_from}")
    return config_to_load

# Load initial config
app_config = load_config()
API_SERVERS = app_config.get('API_SERVERS', []) # Get servers from loaded config
ENABLE_JOB_LOGGING = app_config.get('ENABLE_JOB_LOGGING', True) # Get logging flag

# Define constants used in options loading *before* load_options
MAX_DURATION = 120
DEFAULT_DURATION = 5

# --- Options Loading/Saving ---
OPTIONS_FILE = 'options.json'
DEFAULT_OPTIONS = {
    'prompt': '',
    'duration': DEFAULT_DURATION, # Use constant here too
    'use_teacache': True # Corresponds to NOT using --no-teacache
}

def load_options():
    try:
        if os.path.exists(OPTIONS_FILE):
            with open(OPTIONS_FILE, 'r') as f:
                loaded = json.load(f)
                options = DEFAULT_OPTIONS.copy()
                # Safely update options
                options['prompt'] = str(loaded.get('prompt', DEFAULT_OPTIONS['prompt']))
                try:
                    # Use MAX_DURATION and DEFAULT_DURATION here
                    duration_val = int(loaded.get('duration', DEFAULT_DURATION))
                    options['duration'] = max(1, min(MAX_DURATION, duration_val))
                except (ValueError, TypeError):
                    options['duration'] = DEFAULT_DURATION # Fallback on conversion error
                options['use_teacache'] = bool(loaded.get('use_teacache', DEFAULT_OPTIONS['use_teacache']))
                return options
        else:
            return DEFAULT_OPTIONS.copy()
    except (json.JSONDecodeError, IOError) as e: # Catch IO errors too
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
    global app_options # Allow modification if needed (though saved via API now)
    directory = request.args.get('directory', None) # Get directory from query param
    images = []
    error_message = None

    if request.method == 'POST':
        # Handle directory submission
        submitted_dir = request.form.get('directory')
        if submitted_dir and os.path.isdir(submitted_dir):
            # Valid directory submitted, redirect to GET with query parameter
            return redirect(url_for('index', directory=submitted_dir))
        else:
            # Invalid directory submitted
            error_message = "Invalid or missing directory path."
            # Render the template again, showing the error and the form
            return render_template('index.html', 
                                   directory=None, 
                                   images=[], 
                                   error=error_message, 
                                   options=app_options, 
                                   max_duration=MAX_DURATION)

    # Handle GET request (or after redirect from POST)
    if directory:
        if os.path.isdir(directory):
            try:
                for fname in sorted(os.listdir(directory)): # Sort filenames
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                        images.append({'name': fname, 'path': os.path.join(directory, fname)})
            except OSError as e:
                error_message = f"Error accessing directory: {e}"
                directory = None # Reset directory if error accessing
        else:
            error_message = "Specified directory not found."
            directory = None # Reset directory if invalid

    # Always render index.html, passing current state
    return render_template('index.html', 
                           directory=directory, 
                           images=images, 
                           error=error_message, 
                           options=app_options, 
                           max_duration=MAX_DURATION)

# Remove /view_dir route - functionality merged into '/'
# Remove /options route - replaced by API and integrated into index.html

# --- API Routes ---

@app.route('/api/options', methods=['POST'])
def api_save_options():
    global app_options
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        new_prompt = str(data.get('prompt', app_options['prompt']))
        new_use_teacache = bool(data.get('use_teacache', app_options['use_teacache']))
        
        try:
            new_duration = int(data.get('duration', app_options['duration']))
            if not 1 <= new_duration <= MAX_DURATION:
                 new_duration = app_options['duration'] # Keep old if invalid
                 # Optionally return a warning in the response
        except (ValueError, TypeError):
            new_duration = app_options['duration'] # Keep old on error

        app_options['prompt'] = new_prompt
        app_options['duration'] = new_duration
        app_options['use_teacache'] = new_use_teacache
        
        save_options(app_options)
        
        if ENABLE_JOB_LOGGING:
            logger.info(f"Options updated via API: Prompt='{new_prompt}', Duration={new_duration}, UseTeaCache={new_use_teacache}")
        
        return jsonify({'message': 'Options saved successfully', 'options': app_options}), 200

    except Exception as e:
         if ENABLE_JOB_LOGGING:
             logger.error(f"Error saving options via API: {e}")
         return jsonify({'error': f'Error saving options: {e}'}), 500


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    global app_config, API_SERVERS, server_status
    if request.method == 'GET':
        # Return a copy of the config
        return jsonify(app_config.copy())

    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON payload'}), 400

            new_api_servers_raw = data.get('API_SERVERS')

            # Validate API_SERVERS
            if not isinstance(new_api_servers_raw, list):
                 return jsonify({'error': "'API_SERVERS' must be a list of strings"}), 400
            
            new_api_servers = []
            for server in new_api_servers_raw:
                if not isinstance(server, str) or not server.strip().startswith(('http://', 'https://')):
                    return jsonify({'error': f"Invalid server URL format: '{server}'. Must be strings starting with http:// or https://"}), 400
                new_api_servers.append(server.strip())

            # Update server_status under lock
            with server_lock:
                old_server_status = server_status.copy()
                new_server_status = {}
                for server in new_api_servers:
                    if server in old_server_status:
                        # Preserve status for existing servers
                        new_server_status[server] = old_server_status[server]
                        # Ensure 'available' is True if it wasn't (worker manages busy state)
                        # Let worker manage availability based on jobs and backoff
                        # new_server_status[server]['available'] = True 
                    else:
                        # Initialize new servers
                        new_server_status[server] = {'available': True, 'fail_count': 0, 'next_retry_time': 0}
                        logger.info(f"New API server added: {server}")
                
                # Log removed servers
                removed_servers = set(old_server_status.keys()) - set(new_server_status.keys())
                for server in removed_servers:
                    logger.info(f"API server removed: {server}")

                # Atomically update the global state
                server_status = new_server_status
                API_SERVERS = new_api_servers # Update the list used for worker assignment logic
                app_config['API_SERVERS'] = new_api_servers # Update the config dict

            # Save the updated configuration to file
            save_config(app_config)

            logger.info(f"API_SERVERS configuration updated: {new_api_servers}")
            
            # Note: Changing the number of workers requires an app restart.
            # The current workers will now use the updated API_SERVERS list for job assignment.
            
            return jsonify({'message': 'Configuration saved successfully', 'config': app_config}), 200

        except Exception as e:
             logger.error(f"Error saving configuration via API: {e}")
             return jsonify({'error': f'Error saving configuration: {e}'}), 500


@app.route('/generate', methods=['POST'])
def generate():
    # This route remains largely the same, using app_options as defaults
    global app_options
    image_path = request.form.get('image_path')
    prompt = request.form.get('prompt', app_options['prompt']).strip() 
    duration = request.form.get('duration', app_options['duration'], type=int) 
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
        logger.info(f"Job Submitted - ID: {job_id}, Image: {filename}, Prompt: '{params['prompt']}', Duration: {params['duration']}s, UseTeaCache: {params['use_teacache']}")

    return jsonify({'job_id': job_id, 'status': 'queued'}), 202

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
    # Add argument parsing for port
    parser = argparse.ArgumentParser(description="FramePack Web UI")
    parser.add_argument("--port", type=int, default=5001, help="Port number to run the web server on")
    cli_args = parser.parse_args()

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
        logger.info(f"Starting Flask server with {num_workers} workers on port {cli_args.port}.")
    else:
        print(f"Job logging disabled via config. Starting server on port {cli_args.port}.")

    if ENABLE_JOB_LOGGING:
        logger.info(f"Loaded options: {app_options}")
    else:
        print(f"Loaded options: {app_options}")
        
    # Use the parsed port
    app.run(debug=True, host='0.0.0.0', port=cli_args.port)
