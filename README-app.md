# FramePack Web UI (app.py)

This directory contains a Flask-based web application that provides a user interface for interacting with FramePack generation APIs. It allows users to browse images, queue video generation jobs, and monitor their status.

## Features

*   **Image Browser:** Displays images from a specified directory (`IMAGE_DIR` in `app.py`).
*   **Job Queueing:** Allows users to submit video generation requests via a modal interface.
*   **Concurrent Processing:** Utilizes multiple background worker threads to send requests to different API servers concurrently, configured via `config.json`.
*   **Dynamic Job Status:** A sticky sidebar shows the real-time status of queued, running, completed, and failed jobs.
*   **External Configuration:** API server endpoints are managed in a separate `config.json` file.

## Prerequisites

*   Python 3.x
*   pip (Python package installer)

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/lonestriker/FramePack.git
    cd FramePack
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` includes Flask and any other necessary packages for `app.py`)*

## Configuration

1.  **Copy the example configuration:**
    ```bash
    cp config.json.example config.json
    ```
2.  **Edit `config.json`:**
    Open `config.json` in a text editor and replace the placeholder URLs in the `API_SERVERS` list with the actual URLs of your running FramePack Gradio API instances.
    ```json
    {
      "API_SERVERS": [
        "https://your-actual-api-server-1.com",
        "https://your-actual-api-server-2.com"
        // Add more servers if needed
      ]
    }
    ```
    *This file (`config.json`) is ignored by Git, so your server URLs won't be committed.*

3.  **(Optional) Modify `IMAGE_DIR` in `app.py`:**
    By default, the application serves images from the `framepack/images` directory relative to where `app.py` is run. You can change the `IMAGE_DIR` variable near the top of `app.py` if your images are located elsewhere.

## Running the Application

1.  **Navigate to the directory containing `app.py`:**
    ```bash
    cd /path/to/your/FramePack # Adjust path as necessary
    ```
2.  **Run the Flask development server:**
    ```bash
    python app.py
    ```
3.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000` (or the address provided in the terminal output).

You should now see the web interface, be able to browse images, and submit generation jobs.
