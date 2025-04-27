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
import atexit # For saving queue on exit
import requests # Import requests for sending cancel signal
import sys
import pathlib  # Add pathlib for better path handling
import shlex  # Add shlex for proper command argument escaping
import random

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
MIN_ZOOM = 80
MAX_ZOOM = 300
DEFAULT_ZOOM = 150

# --- Options Loading/Saving ---
OPTIONS_FILE = 'options.json'
DEFAULT_OPTIONS = {
    'prompt': '',
    'duration': DEFAULT_DURATION, 
    'use_teacache': True,
    'zoom_level': DEFAULT_ZOOM # Add zoom level
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
                    duration_val = int(loaded.get('duration', DEFAULT_DURATION))
                    options['duration'] = max(1, min(MAX_DURATION, duration_val))
                except (ValueError, TypeError):
                    options['duration'] = DEFAULT_DURATION 
                options['use_teacache'] = bool(loaded.get('use_teacache', DEFAULT_OPTIONS['use_teacache']))
                try: # Load and validate zoom level
                    zoom_val = int(loaded.get('zoom_level', DEFAULT_ZOOM))
                    options['zoom_level'] = max(MIN_ZOOM, min(MAX_ZOOM, zoom_val))
                except (ValueError, TypeError):
                     options['zoom_level'] = DEFAULT_ZOOM # Fallback on conversion error
                return options
        else:
            return DEFAULT_OPTIONS.copy()
    except (json.JSONDecodeError, IOError) as e: 
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

# --- Queue State Persistence ---
QUEUE_STATE_FILE = 'queue_state.json'

def save_queue_state():
    """Saves non-final jobs (queued, failed_will_retry) to a file."""
    try:
        state_to_save = {}
        # Create a copy to avoid modifying dict during iteration
        current_jobs = dict(job_status) 
        for job_id, data in current_jobs.items():
            if data.get('status') in ['queued', 'failed_will_retry']:
                state_to_save[job_id] = data
        
        with open(QUEUE_STATE_FILE, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        if ENABLE_JOB_LOGGING:
            logger.info(f"Saved {len(state_to_save)} pending jobs to {QUEUE_STATE_FILE}")
            
    except Exception as e:
        print(f"Error saving queue state: {e}")
        if ENABLE_JOB_LOGGING:
            logger.error(f"Failed to save queue state to {QUEUE_STATE_FILE}: {e}")

def load_queue_state():
    """Loads queue state from file and re-populates the queue."""
    global queue_paused # Allow modification
    jobs_loaded = 0
    if os.path.exists(QUEUE_STATE_FILE):
        try:
            with open(QUEUE_STATE_FILE, 'r') as f:
                loaded_state = json.load(f)
            
            if loaded_state:
                print(f"Loading {len(loaded_state)} pending jobs from {QUEUE_STATE_FILE}...")
                for job_id, data in loaded_state.items():
                    if job_id not in job_status: # Avoid overwriting if somehow already present
                        job_status[job_id] = data
                        # Re-add to the actual queue for processing
                        job_queue.put((job_id, data['params'])) 
                        jobs_loaded += 1
                
                if jobs_loaded > 0:
                    print(f"Restored {jobs_loaded} jobs. Pausing queue on startup.")
                    logger.info(f"Restored {jobs_loaded} jobs from {QUEUE_STATE_FILE}. Pausing queue.")
                    queue_paused = True # Pause queue by default if jobs were restored
                else:
                     print("Queue state file found, but no pending jobs to restore.")
                     logger.info("Queue state file found, but no pending jobs to restore.")
            else:
                 print(f"{QUEUE_STATE_FILE} is empty. No jobs to restore.")
                 logger.info(f"{QUEUE_STATE_FILE} is empty. No jobs to restore.")

        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Error loading queue state from {QUEUE_STATE_FILE}: {e}. Starting with empty queue.")
            logger.error(f"Error loading queue state from {QUEUE_STATE_FILE}: {e}. Starting with empty queue.")
            # Optionally delete or rename the corrupted file
            # os.remove(QUEUE_STATE_FILE) 
    else:
        print("No queue state file found. Starting with empty queue.")
        logger.info("No queue state file found. Starting with empty queue.")
    return jobs_loaded > 0 # Return true if queue was paused

# --- In-memory state (for simplicity, replace with DB/proper state management later) ---
job_queue = queue.Queue()
# Updated job_status structure
job_status = {} # {job_id: {'status': 'queued'/'running'/'completed'/'failed_will_retry', 'output': [], 'params': {}, 'creation_time': '...', 'cancelled': False, 'progress': 0, 'message': ''}} # Added progress/message
# Replace active_servers with server_status
server_status = {
    server: {'available': True, 'fail_count': 0, 'next_retry_time': 0, 'enabled': True} # Add enabled flag
    for server in API_SERVERS
}
server_lock = threading.Lock() # Keep the lock
queue_paused = False # Initialize queue pause state

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
        if (submitted_dir and os.path.isdir(submitted_dir)):
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
        # Use pathlib for better path handling
        dir_path = pathlib.Path(directory)
        if dir_path.is_dir():
            try:
                for fname in sorted(os.listdir(dir_path)): 
                    file_path = dir_path / fname
                    if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                        # Ensure path is converted to string with forward slashes for URLs
                        url_path = str(file_path).replace('\\', '/')
                        images.append({'name': fname, 'path': url_path})
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
                 new_duration = app_options['duration'] 
        except (ValueError, TypeError):
            new_duration = app_options['duration'] 

        try: # Validate zoom level from payload
            new_zoom = int(data.get('zoom_level', app_options['zoom_level']))
            if not MIN_ZOOM <= new_zoom <= MAX_ZOOM:
                 new_zoom = app_options['zoom_level'] # Keep old if invalid
        except (ValueError, TypeError):
            new_zoom = app_options['zoom_level'] # Keep old on error

        app_options['prompt'] = new_prompt
        app_options['duration'] = new_duration
        app_options['use_teacache'] = new_use_teacache
        app_options['zoom_level'] = new_zoom # Update zoom level
        
        save_options(app_options)
        
        if ENABLE_JOB_LOGGING:
            logger.info(f"Options updated via API: Prompt='{new_prompt}', Duration={new_duration}, UseTeaCache={new_use_teacache}, Zoom={new_zoom}")
        
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
                        new_server_status[server] = {'available': True, 'fail_count': 0, 'next_retry_time': 0, 'enabled': True}
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

    try:
        # Convert to Path object
        path_obj = pathlib.Path(image_path)
        normalized_image_path = str(path_obj.resolve())
        
        if not path_obj.is_file():
            print(f"Path exists but is not a file: {normalized_image_path}")
            return jsonify({'error': 'Path exists but is not a file'}), 400
    except (TypeError, ValueError) as e:
        print(f"Error normalizing path: {e}")
        return jsonify({'error': f'Invalid image path: {e}'}), 400
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return jsonify({'error': 'File not found'}), 400
    
    if not os.path.isfile(normalized_image_path):
        print(f"Normalized path check failed: {normalized_image_path}")
        return jsonify({'error': 'Invalid image path after normalization'}), 400

    # Validate duration
    if not 1 <= duration <= MAX_DURATION:
        return jsonify({'error': f'Invalid duration. Must be between 1 and {MAX_DURATION}'}), 400

    job_id = str(uuid.uuid4())
    params = {
        'image_path': normalized_image_path,
        'prompt': prompt,
        'duration': duration,
        'use_teacache': use_teacache_for_job # Store the effective setting for this job
    }
    # Add creation_time, cancelled flag, progress, and message
    job_status[job_id] = {
        'status': 'queued', 
        'output': [], 
        'params': params, 
        'creation_time': datetime.utcnow().isoformat() + 'Z', # Add UTC timestamp
        'cancelled': False, # Initialize cancelled flag
        'progress': 0,      # Initialize progress
        'message': 'Queued' # Initialize message
    }
    job_queue.put((job_id, params))
    save_queue_state() # Save queue state after adding a job

    if ENABLE_JOB_LOGGING:
        filename = os.path.basename(image_path)
        logger.info(f"Job Submitted - ID: {job_id}, Image: {filename}, Prompt: '{params['prompt']}', Duration: {params['duration']}s, UseTeaCache: {params['use_teacache']}")

    return jsonify({'job_id': job_id, 'status': 'queued'}), 202

# --- New API Route for Job Status ---
@app.route('/api/jobs')
def api_jobs():
    # Return a copy to avoid issues if dict changes during serialization
    # Ensure the cancelled flag is included if needed by frontend logic later
    return jsonify(dict(job_status)) 

# --- New API Route for Cancelling Jobs ---
@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404

    job_info = job_status[job_id]
    current_status = job_info.get('status')
    
    if current_status in ['completed', 'failed', 'cancelled', 'cancelling']:
         return jsonify({'message': f'Job already in final or cancelling state: {current_status}'}), 400

    if current_status in ['queued', 'failed_will_retry']:
        job_info['status'] = 'cancelled'
        job_info['cancelled'] = True # Set the flag
        job_info['output'].append("Job cancelled by user before execution.")
        logger.info(f"Job Cancelled (Queued) - ID: {job_id}")
        save_queue_state() # Save state after cancelling queued job
        return jsonify({'message': 'Job cancelled successfully'}), 200
        
    elif current_status == 'running':
        job_info['status'] = 'cancelling' # Mark as cancelling
        job_info['cancelled'] = True # Set the flag for the worker to potentially see (best effort)
        job_info['output'].append("Cancellation requested by user...")
        logger.info(f"Job Cancellation Requested (Running) - ID: {job_id}")
        
        # --- Send cancel signal to the assigned API server ---
        assigned_server = job_info.get('assigned_server')
        if assigned_server:
            cancel_url = f"{assigned_server}/cancel-current"
            logger.info(f"Sending cancel signal to {cancel_url} for job {job_id}")
            try:
                # Send request in a separate thread to avoid blocking Flask? Or use async client?
                # For simplicity, using a blocking request with a short timeout for now.
                # Consider using httpx or similar for async requests if this becomes a bottleneck.
                response = requests.post(cancel_url, timeout=5) # 5 second timeout
                if response.status_code == 200:
                    logger.info(f"Successfully sent cancel signal to {assigned_server} for job {job_id}. Server response: {response.text}")
                    job_info['output'].append("Cancel signal sent to processing server.")
                elif response.status_code == 404:
                     logger.warning(f"Cancel signal sent, but server {assigned_server} reported no active job (maybe finished?). Job: {job_id}")
                     job_info['output'].append("Cancel signal sent, server reported no active job.")
                else:
                    logger.error(f"Failed to send cancel signal to {assigned_server} for job {job_id}. Status code: {response.status_code}, Response: {response.text}")
                    job_info['output'].append(f"Error sending cancel signal to server (HTTP {response.status_code}).")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error sending cancel signal to {assigned_server} for job {job_id}: {e}")
                job_info['output'].append(f"Error communicating with server to send cancel signal: {e}")
        else:
            logger.warning(f"Cannot send cancel signal for job {job_id}: Assigned server not found in job status.")
            job_info['output'].append("Could not determine server to send cancel signal.")
            
        # Note: save_queue_state will be called by the worker when it finishes/fails/cancels
        return jsonify({'message': 'Cancellation requested. Signal sent to processing server (best effort).'}), 200
    else:
        # Should not happen based on initial check, but good practice
        return jsonify({'error': f'Cannot cancel job in state: {current_status}'}), 400

# --- Queue Control API Routes ---
@app.route('/api/queue/clear', methods=['POST'])
def clear_queue():
    global job_status, job_queue
    cleared_count = 0
    
    # Need to rebuild the queue and job_status carefully
    new_job_status = {}
    temp_queue_items = []

    # Drain the current queue first
    while not job_queue.empty():
        try:
            temp_queue_items.append(job_queue.get_nowait())
        except queue.Empty:
            break
            
    # Iterate through job_status and keep non-queued/non-retry jobs
    for job_id, data in job_status.items():
        if data.get('status') not in ['queued', 'failed_will_retry']:
            new_job_status[job_id] = data
        else:
            cleared_count += 1
            logger.info(f"Clearing job {job_id} with status {data.get('status')}")

    # Re-populate the queue only with items that were NOT cleared from job_status
    # This handles the case where a job might be in the queue but already marked differently in status
    # (though ideally status and queue are consistent)
    for item_job_id, item_params in temp_queue_items:
        if item_job_id in new_job_status: # Should only happen if status wasn't queued/retry
             job_queue.put((item_job_id, item_params))
             
    job_status = new_job_status # Atomically update job_status
    
    logger.info(f"Cleared {cleared_count} jobs from the queue.")
    save_queue_state() # Save the now empty/reduced queue state
    return jsonify({'message': f'Cleared {cleared_count} jobs from the queue.'}), 200

@app.route('/api/queue/pause', methods=['POST'])
def pause_queue():
    global queue_paused
    queue_paused = True
    logger.info("Queue paused.")
    return jsonify({'message': 'Queue paused successfully.', 'status': 'paused'}), 200

@app.route('/api/queue/run', methods=['POST'])
def run_queue():
    global queue_paused
    queue_paused = False
    logger.info("Queue resumed.")
    return jsonify({'message': 'Queue resumed successfully.', 'status': 'running'}), 200

@app.route('/api/queue/status', methods=['GET'])
def get_queue_status():
    return jsonify({'status': 'paused' if queue_paused else 'running'}), 200

# --- Server Control API Routes ---
@app.route('/api/servers/<path:server_url>/toggle', methods=['POST'])
def toggle_server(server_url):
    # Note: server_url might need decoding if it contains special characters, 
    # but Flask usually handles basic URL path decoding. Test this.
    # For simplicity, assume basic URLs for now.
    
    with server_lock:
        if server_url not in server_status:
            # If server was added via config UI but app not restarted, it might not be here yet.
            # Or if URL is mangled.
            # Check against API_SERVERS list as well.
            if server_url not in API_SERVERS:
                 logger.warning(f"Attempted to toggle unknown server: {server_url}")
                 return jsonify({'error': 'Server not found in current configuration'}), 404
            else:
                 # Initialize status for a server known in config but not yet in runtime status
                 server_status[server_url] = {'available': True, 'fail_count': 0, 'next_retry_time': 0, 'enabled': True}

        current_state = server_status[server_url].get('enabled', True) # Default to enabled if key missing
        new_state = not current_state
        server_status[server_url]['enabled'] = new_state
        
        # Reset availability/backoff when enabling? Maybe not, let it recover naturally.
        # if new_state:
        #     server_status[server_url]['available'] = True
        #     server_status[server_url]['next_retry_time'] = 0
            
        action = "enabled" if new_state else "disabled"
        logger.info(f"Server {server_url} {action}.")
        
    # No need to save config, this is runtime state
    return jsonify({'message': f'Server {server_url} {action}.', 'server': server_url, 'enabled': new_state}), 200


@app.route('/api/server_status')
def api_server_status():
    with server_lock:
        status_copy = {}
        current_time = time.time()
        # Use API_SERVERS as the source of truth for which servers *should* exist
        for server in API_SERVERS:
            status = server_status.get(server, {'available': False, 'fail_count': 0, 'next_retry_time': 0, 'enabled': False}) # Default if missing
            status_copy[server] = status.copy() # Create a copy of the inner dict
            status_copy[server]['next_retry_available_in_seconds'] = max(0, status['next_retry_time'] - current_time)
            status_copy[server]['next_retry_time_readable'] = time.ctime(status['next_retry_time']) if status['next_retry_time'] > 0 else "N/A"
            # Ensure enabled status is included, defaulting to True if somehow missing
            status_copy[server]['enabled'] = status.get('enabled', True) 
        return jsonify(status_copy)

# --- Route to serve images safely ---
@app.route('/images/<path:filename>')
def serve_image(filename):
    # IMPORTANT: Add security checks here in a real application 
    # to prevent access outside allowed directories.
    
    try:
        # Use pathlib for better path handling
        path_obj = pathlib.Path(filename)
        
        # Check if the file exists directly
        if path_obj.is_file():
            return send_file(str(path_obj))
        
        # If not found, try looking in the working directory
        current_dir = pathlib.Path(__file__).parent.absolute()
        potential_path = current_dir / filename
        
        if potential_path.is_file():
            return send_file(str(potential_path))
            
        # If still not found, return 404
        print(f"Image not found: {filename} (path_obj: {path_obj}, potential: {potential_path})")
        return "Image not found", 404
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        if ENABLE_JOB_LOGGING:
            logger.error(f"Error serving image {filename}: {e}")
        return "Error serving image", 500

# --- Route to serve results safely ---
@app.route('/results/<path:filename>')
def serve_result(filename):
    # IMPORTANT: Add security checks here in a real application 
    # Ensure filename is safe and only accesses the intended output directory
    results_dir = os.path.abspath('./downloads') # Or wherever api-client saves files
    file_path = os.path.abspath(os.path.join(results_dir, filename))

    # Prevent path traversal attacks
    if not file_path.startswith(results_dir):
        logger.warning(f"Attempted access outside results directory: {filename}")
        return "Forbidden", 403
        
    try:
        if not os.path.exists(file_path):
             logger.error(f"Result file not found: {filename}")
             return "Result file not found", 404
        logger.info(f"Serving result file: {filename}")
        return send_file(file_path, as_attachment=False) # Serve inline if possible
    except Exception as e:
        logger.error(f"Error serving result file {filename}: {e}")
        return "Error serving file", 500

# --- Background Worker ---

def worker():
    while True:
        # --- Check Pause State ---
        while queue_paused:
            time.sleep(1) # Wait if paused

        try:
            # Use timeout to allow checking pause state periodically
            job_id, params = job_queue.get(timeout=1) 
        except queue.Empty:
            continue # No job, loop back to check pause state

        # --- Check for Cancellation BEFORE Assignment ---
        if job_status.get(job_id, {}).get('cancelled', False) or job_status.get(job_id, {}).get('status') == 'cancelled':
            if job_status.get(job_id, {}).get('status') != 'cancelled': # Ensure status is set if only flag was true
                job_status[job_id]['status'] = 'cancelled'
                job_status[job_id]['output'].append("Job cancelled before server assignment.")
                logger.info(f"Worker found job {job_id} cancelled before assignment.")
            job_queue.task_done()
            continue # Skip to the next job

        # --- Server Selection ---
        assigned_server = None
        while assigned_server is None:
            # Check pause state again within the server selection loop
            if queue_paused:
                 print(f"Queue paused while selecting server for job {job_id}. Re-queuing.")
                 logger.info(f"Queue paused while selecting server for job {job_id}. Re-queuing.")
                 job_queue.put((job_id, params)) # Put job back
                 assigned_server = "paused" # Special value to break loop
                 break

            current_time = time.time()
            with server_lock:
                # Find the best available server (least recently failed or ready for retry)
                candidates = []
                for server, status in server_status.items():
                    # Add check for 'enabled' status
                    if status.get('enabled', True) and status['available'] and current_time >= status['next_retry_time']:
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
                     # Add a check here too in case it was cancelled while waiting for a server
                     if job_status.get(job_id, {}).get('cancelled', False):
                         job_status[job_id]['status'] = 'cancelled'
                         job_status[job_id]['output'].append("Job cancelled while waiting for server.")
                         logger.info(f"Worker found job {job_id} cancelled while waiting for server.")
                         job_queue.task_done()
                         assigned_server = "cancelled" # Special value to break outer loop
                         break # Break inner server selection loop
                     time.sleep(0.5) # Wait before checking again
        
        if assigned_server in ["cancelled", "paused"]:
            # If paused, job was already re-queued. If cancelled, just continue.
            job_queue.task_done() # Mark task done even if paused/cancelled here
            continue # Skip to next job in the main worker loop

        # --- Record Start Time ---
        start_time = time.time()
        job_status[job_id]['start_time'] = start_time
        # --- End Record Start Time ---

        # --- Job Execution ---
        job_status[job_id]['status'] = 'running'
        job_status[job_id]['progress'] = 0 # Reset progress when starting
        job_status[job_id]['message'] = 'Starting execution...' # Update message
        job_status[job_id]['start_time'] = datetime.utcnow().isoformat() + 'Z'  # ISO format with UTC timezone
        job_status[job_id]['assigned_server'] = assigned_server # Store the assigned server

        job_failed = False
        error_message = ""

        try:
            # Use pathlib for better path handling
            image_path = pathlib.Path(params['image_path'])
            
            # Prepare command with proper path handling
            command = [
                sys.executable,  # Use the current Python interpreter
                'api-client.py',
                '--api_url', assigned_server,
                '--prompt', params['prompt'],
                '--length', str(params['duration']),
                '--image', str(image_path),  # Convert Path object to string
                '--seed', str(random.randint(0, 2**32 - 1)), # Random seed for each job
            ]
            if not params.get('use_teacache', True): # Default to True if missing
                command.append('--no-teacache')

            # Log the actual command being run
            print(f"Executing command for job {job_id}: {' '.join(shlex.quote(str(arg)) for arg in command)}")
            if ENABLE_JOB_LOGGING:
                logger.info(f"Executing Command - ID: {job_id}, Server: {assigned_server}, Image: {image_path.name}, Command: {' '.join(shlex.quote(str(arg)) for arg in command)}")

            process = subprocess.Popen(command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       text=True,
                                       bufsize=1,
                                       universal_newlines=True,
                                       cwd=str(pathlib.Path(__file__).parent.absolute()))

            output_lines = []
            # Note: We don't have a reliable way to check the 'cancelled' flag *during* Popen
            # without more complex async process handling. Cancellation for running jobs
            # is primarily handled by setting the status to 'cancelling' via the API.
            for line in process.stdout:
                stripped_line = line.strip()
                
                # --- Check for Progress Updates ---
                if stripped_line.startswith("PROGRESS::"):
                    try:
                        parts = stripped_line.split("::", 2)
                        if len(parts) == 3:
                            progress_val = int(parts[1])
                            message_val = parts[2]
                            job_status[job_id]['progress'] = progress_val
                            job_status[job_id]['message'] = message_val
                            # Optionally log progress updates
                            # if ENABLE_JOB_LOGGING:
                            #     logger.debug(f"Job {job_id} Progress: {progress_val}% - {message_val}")
                        continue # Don't add progress lines to regular output
                    except (ValueError, IndexError):
                        # Ignore malformed progress lines
                        pass 
                # --- End Progress Update Check ---

                # Store regular output lines
                if len(job_status[job_id]['output']) < 100: # Limit output lines
                    job_status[job_id]['output'].append(stripped_line)
                output_lines.append(stripped_line) # Keep local copy for logging

            process.wait()

            # Check status *after* process finishes
            if job_status.get(job_id, {}).get('status') == 'cancelling':
                 # If it was marked cancelling, update to cancelled now that process finished
                 job_status[job_id]['status'] = 'cancelled'
                 job_status[job_id]['output'].append("Job process finished after cancellation request.")
                 logger.warning(f"Job Finished After Cancel - ID: {job_id}, Server: {assigned_server}. Final RC: {process.returncode}")
                 job_failed = True # Treat as failure for server backoff purposes? Or just cancelled? Let's treat as cancelled.
                 job_failed = False # Override: Treat as cancelled, don't penalize server unless it actually errored.
                 error_message = "Job cancelled by user." # Set message for logging/status
                 # Skip normal success/fail logic below if cancelled
            elif process.returncode != 0:
                job_failed = True
                # More specific error message
                error_message = f"Job script failed on {assigned_server} with return code: {process.returncode}"
                job_status[job_id]['output'].append(error_message)
                job_status[job_id]['message'] = f"Failed (RC: {process.returncode})" # Update message on failure
                job_status[job_id]['progress'] = 0 # Reset progress on failure? Or leave as is? Resetting.
            else:
                # Success!
                job_status[job_id]['status'] = 'completed'
                job_status[job_id]['output'].append("Job completed successfully.")
                job_status[job_id]['progress'] = 100 # Set progress to 100 on completion
                job_status[job_id]['message'] = "Completed successfully" # Update message
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
            # Check if cancelled during exception handling
            if job_status.get(job_id, {}).get('status') == 'cancelling':
                 job_status[job_id]['status'] = 'cancelled'
                 job_status[job_id]['output'].append(f"Job exception occurred after cancellation request: {e}")
                 logger.warning(f"Job Exception After Cancel - ID: {job_id}, Server: {assigned_server}. Error: {e}")
                 job_failed = False # Treat as cancelled
                 error_message = "Job cancelled by user (exception occurred)."
            else:
                 job_failed = True
                 error_message = f"Exception during job execution on {assigned_server}: {e}"
                 job_status[job_id]['output'].append(error_message)
                 job_status[job_id]['message'] = f"Exception: {e}" # Update message on exception
                 job_status[job_id]['progress'] = 0 # Reset progress
                 print(f"Job {job_id} encountered exception on {assigned_server}: {e}")

        finally:
            # --- Record End Time & Calculate Duration ---
            end_time = time.time()
            job_info = job_status.get(job_id, {})
            recorded_start_time = job_info.get('start_time')
            if recorded_start_time:
                try:
                    # Convert ISO string timestamp to datetime object, then to timestamp
                    if isinstance(recorded_start_time, str):
                        # Handle ISO format correctly
                        if recorded_start_time.endswith('Z'):
                            # Replace 'Z' with '+00:00' for proper UTC parsing
                            recorded_start_time = recorded_start_time.replace('Z', '+00:00')
                        start_time_dt = datetime.fromisoformat(recorded_start_time)
                        start_timestamp = start_time_dt.timestamp()
                    else:
                        # Already a timestamp (float)
                        start_timestamp = recorded_start_time
                    
                    duration_seconds = end_time - start_timestamp
                    # Sanity check for duration
                    if duration_seconds < 0 or duration_seconds > 3600:  # More than an hour is likely wrong
                        logger.warning(f"Suspicious duration for job {job_id}: {duration_seconds:.2f}s. Using elapsed time from job start.")
                        # Fallback to elapsed time since worker started processing
                        start_time = job_status[job_id].get('start_time', end_time)
                        if isinstance(start_time, (int, float)):
                            duration_seconds = end_time - start_time
                        else:
                            duration_seconds = 0  # Failed to get a valid duration
                            
                    job_info['generation_duration_seconds'] = duration_seconds
                    # Optionally log the duration
                    if ENABLE_JOB_LOGGING:
                        logger.info(f"Job {job_id} execution duration: {duration_seconds:.2f} seconds.")
                except (ValueError, TypeError) as e:
                    # Handle parsing errors
                    logger.error(f"Error calculating duration for job {job_id}: {e}")
                    # Use a fallback duration if needed
                    job_info['generation_duration_seconds'] = 0
            # --- End Duration Calculation ---

            # --- Server Release / Re-queue Logic ---
            server_needs_release = True 
            # Check final status - if it ended up as 'cancelled' or 'cancelling', don't re-queue
            final_status = job_status.get(job_id, {}).get('status')
            
            if job_failed and final_status not in ['cancelled', 'cancelling']:
                # Failure: Penalize server, re-queue job, mark server available for future (after backoff)
                job_status[job_id]['status'] = 'failed_will_retry' 
                job_status[job_id]['output'].append(f"Job failed on {assigned_server}. Re-queuing.")
                job_status[job_id]['message'] = f"Failed, will retry. ({error_message})" # Update message
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
            elif final_status in ['cancelled', 'cancelling']:
                 # Job was cancelled, don't re-queue. Release server normally (no penalty).
                 logger.info(f"Job {job_id} ended with status {final_status}. Server {assigned_server} will be released without penalty.")
                 job_failed = False # Ensure it's not treated as a failure for release logic
                 server_needs_release = True # Ensure server is released below
                 if final_status == 'cancelling': # If it was still 'cancelling', mark as 'cancelled'
                     job_status[job_id]['status'] = 'cancelled'
                     job_status[job_id]['message'] = "Cancelled by user"

            # --- Simplified Server Release Logic ---
            if server_needs_release:
                 with server_lock:
                      # Check if the server still exists in the status dict (might have been removed by config change)
                      if assigned_server in server_status:
                          # Always mark available and reset state if job didn't fail
                          server_status[assigned_server]['available'] = True
                          server_status[assigned_server]['fail_count'] = 0 
                          server_status[assigned_server]['next_retry_time'] = 0
                          print(f"Server {assigned_server} released (job success/cancelled).")
                      else:
                          print(f"Server {assigned_server} no longer in config, cannot release.")
            # --- End Simplified Server Release Logic ---

            # Save queue state after job completion/failure/cancellation
            save_queue_state() 

            job_queue.task_done() # Signal task completion for queue management

# --- Main Execution ---
if __name__ == '__main__':
    # Add argument parsing for port
    parser = argparse.ArgumentParser(description="FramePack Web UI")
    parser.add_argument("--port", type=int, default=5000, help="Port number to run the web server on")
    cli_args = parser.parse_args()

    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Load initial queue state BEFORE starting workers
    was_paused_on_load = load_queue_state()

    # Register saving queue state on exit
    atexit.register(save_queue_state)
        
    # Start background worker thread(s)
    # Match the number of workers to the number of API servers for concurrency
    num_workers = len(API_SERVERS)
    print(f"Starting {num_workers} worker threads...")
    for i in range(num_workers):
        thread = threading.Thread(target=worker, daemon=True, name=f"Worker-{i+1}")
        thread.start()

    if ENABLE_JOB_LOGGING:
        logger.info(f"Starting Flask server with {num_workers} workers on port {cli_args.port}.")
        if was_paused_on_load:
             logger.info("Queue was paused on startup due to restored jobs.")
    else:
        print(f"Job logging disabled via config. Starting server on port {cli_args.port}.")
        if was_paused_on_load:
             print("Queue was paused on startup due to restored jobs.")

    if ENABLE_JOB_LOGGING:
        logger.info(f"Loaded options: {app_options}")
    else:
        print(f"Loaded options: {app_options}")
        
    # Use the parsed port
    app.run(debug=True, host='0.0.0.0', port=cli_args.port)
