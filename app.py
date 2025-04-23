from flask import Flask, render_template, request, url_for, send_file, jsonify, redirect
import os
import threading
import queue
import subprocess
import uuid
import time
import json

app = Flask(__name__)

# --- Configuration Loading ---
def load_config():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            if not isinstance(config.get('API_SERVERS'), list):
                raise ValueError("'API_SERVERS' key missing or not a list in config.json")
            return config
    except FileNotFoundError:
        print("ERROR: config.json not found. Please create it based on config.json.example.")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Could not parse config.json: {e}")
        exit(1)

config_data = load_config()
API_SERVERS = config_data.get('API_SERVERS', []) # Get servers from loaded config

# Configuration Constants (can still be defined here if needed)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
MAX_DURATION = 120
DEFAULT_DURATION = 5

# --- In-memory state (for simplicity, replace with DB/proper state management later) ---
job_queue = queue.Queue()
job_status = {} # {job_id: {'status': 'queued'/'running'/'completed'/'failed', 'output': [], 'params': {}}}
active_servers = {server: True for server in API_SERVERS} # Track server availability
server_lock = threading.Lock()

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
                           default_duration=DEFAULT_DURATION)

@app.route('/generate', methods=['POST'])
def generate():
    image_path = request.form.get('image_path')
    prompt = request.form.get('prompt', '')
    duration = request.form.get('duration', DEFAULT_DURATION, type=int)

    if not image_path or not os.path.isfile(image_path):
        return jsonify({'error': 'Invalid image path'}), 400

    # Validate duration
    if not 1 <= duration <= MAX_DURATION:
         # Use default or return error, returning error is clearer for API
        return jsonify({'error': f'Invalid duration. Must be between 1 and {MAX_DURATION}'}), 400

    job_id = str(uuid.uuid4())
    params = {
        'image_path': image_path,
        'prompt': prompt,
        'duration': duration,
    }
    job_status[job_id] = {'status': 'queued', 'output': [], 'params': params}
    job_queue.put((job_id, params))

    return jsonify({'job_id': job_id, 'status': 'queued'}), 202 # 202 Accepted

# --- New API Route for Job Status ---
@app.route('/api/jobs')
def api_jobs():
    # Return a copy to avoid issues if dict changes during serialization
    return jsonify(dict(job_status))

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
        
        assigned_server = None
        while assigned_server is None:
            with server_lock:
                for server, available in active_servers.items():
                    if available:
                        active_servers[server] = False # Mark as busy
                        assigned_server = server
                        break
            if assigned_server is None:
                # If no server is free, wait a bit before checking again
                time.sleep(1) 

        job_status[job_id]['status'] = 'running'
        job_status[job_id]['output'].append(f"Assigned to server: {assigned_server}")
        
        try:
            command = [
                'python',
                'api-client.py',
                '--api_url', assigned_server,
                '--prompt', params['prompt'],
                '--length', str(params['duration']), 
                '--image', params['image_path']
            ]
            
            # Using subprocess.Popen for real-time output capture
            process = subprocess.Popen(command, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.STDOUT, # Combine stdout and stderr
                                       text=True, 
                                       bufsize=1, # Line buffered
                                       universal_newlines=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))) # Run in script's dir

            # Capture output line by line
            for line in process.stdout:
                job_status[job_id]['output'].append(line.strip())
            
            process.wait() # Wait for the process to complete

            if process.returncode == 0:
                job_status[job_id]['status'] = 'completed'
                job_status[job_id]['output'].append("Job completed successfully.")
            else:
                job_status[job_id]['status'] = 'failed'
                job_status[job_id]['output'].append(f"Job failed with return code: {process.returncode}")

        except Exception as e:
            job_status[job_id]['status'] = 'failed'
            job_status[job_id]['output'].append(f"Error executing job: {e}")
        finally:
             # Release the server
            with server_lock:
                 active_servers[assigned_server] = True
            job_queue.task_done()

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

    app.run(debug=True, host='0.0.0.0') # Run on all available interfaces
