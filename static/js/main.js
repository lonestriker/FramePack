// Frontend JavaScript for FramePack UI

document.addEventListener('DOMContentLoaded', () => {
    console.log('FramePack UI Initialized');

    const imageModalElement = document.getElementById('imageModal');
    const imageModal = new bootstrap.Modal(imageModalElement);
    const modalImagePreview = document.getElementById('modalImagePreview');
    const modalImageName = document.getElementById('modalImageName');
    const modalImagePathInput = document.getElementById('modalImagePath');
    const generateForm = document.getElementById('generateForm');
    const modalMessage = document.getElementById('modalMessage');
    const activeJobsList = document.getElementById('active-jobs-list');
    const finishedJobsList = document.getElementById('finished-jobs-list');
    const durationInput = document.getElementById('duration'); // Assuming duration input has id='duration'

    // --- Modal Handling --- 
    imageModalElement.addEventListener('show.bs.modal', (event) => {
        // Button that triggered the modal
        const button = event.relatedTarget;
        // Extract info from data-* attributes
        const imagePath = button.getAttribute('data-image-path');
        const imageName = button.getAttribute('data-image-name');
        const imageSrc = button.getAttribute('data-image-src');
        
        // Update the modal's content.
        modalImagePreview.src = imageSrc;
        modalImagePreview.alt = imageName;
        modalImageName.textContent = imageName;
        modalImagePathInput.value = imagePath;
        
        // Reset form state
        generateForm.reset(); // Clear previous prompt etc.
        // Duration input likely has default value set in HTML, check if needed
        // durationInput.value = defaultDuration; // If needed & passed via data-* on body
        modalMessage.textContent = ''; // Clear previous messages
    });

    // --- Form Submission (Generate Job) ---
    generateForm.addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent traditional form submission
        modalMessage.textContent = 'Submitting job...';

        const formData = new FormData(generateForm);

        fetch('/generate', { // Use the correct endpoint
            method: 'POST',
            body: formData // FormData handles content type automatically
        })
        .then(response => {
            if (!response.ok) {
                 // Try to parse error message from backend JSON
                 return response.json().then(err => {
                    throw new Error(err.error || `HTTP error! Status: ${response.status}`);
                }).catch(() => {
                    // Fallback if parsing fails or no JSON body
                    throw new Error(`HTTP error! Status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Job submitted:', data);
            modalMessage.textContent = `Job ${data.job_id} queued successfully!`;
            // Optionally close modal after a short delay
            setTimeout(() => {
                imageModal.hide();
                fetchJobStatus(); // Update queue immediately
            }, 1500);
        })
        .catch(error => {
            console.error('Error submitting job:', error);
            modalMessage.textContent = `Error: ${error.message}`; 
        });
    });

    // --- Job Status Polling and Display ---
    function formatJobStatus(jobId, job) {
        let statusClass = 'list-group-item-secondary'; // Default (queued)
        let statusText = job.status.toUpperCase();
        let badgeClass = 'bg-secondary';
        let outputLog = job.output ? job.output.join('\n') : 'No output yet.';

        if (job.status === 'running') {
            statusClass = 'list-group-item-info';
            badgeClass = 'bg-info text-dark';
        } else if (job.status === 'completed') {
            statusClass = 'list-group-item-success';
            badgeClass = 'bg-success';
        } else if (job.status === 'failed') {
            statusClass = 'list-group-item-danger';
            badgeClass = 'bg-danger';
        }

        // Basic display - shows status and job ID
        // TODO: Improve display, maybe add button for logs?
        let imageName = job.params?.image_path?.split(/[\\/]/).pop() || 'N/A'; // Extract filename
        return `
            <a href="#" class="list-group-item list-group-item-action ${statusClass}" aria-current="true" title="${outputLog}">
                <div class="d-flex w-100 justify-content-between">
                    <small class="mb-1">${jobId.substring(0, 8)}...</small>
                    <span class="badge ${badgeClass} rounded-pill">${statusText}</span>
                </div>
                <p class="mb-1"><small>${imageName}</small></p>
                 ${job.params?.prompt ? `<p class="mb-1"><small>Prompt: ${job.params.prompt.substring(0,30)}...</small></p>` : ''}
            </a>
        `;
         // Consider adding a button here to show logs in a separate modal later
    }

    function updateJobQueue(jobs) {
        if (!activeJobsList || !finishedJobsList) return; // Check for new list elements

        let activeHtml = '';
        let finishedHtml = '';
        let activeCount = 0;
        let finishedCount = 0;

        // Sort jobs by status first (e.g., running, queued, completed, failed)
        // Then potentially by timestamp if available (requires adding timestamp to job_status)
        const sortedJobIds = Object.keys(jobs).sort((a, b) => {
            const statusOrder = { 'running': 1, 'queued': 2, 'completed': 3, 'failed': 4 };
            const statusA = statusOrder[jobs[a].status] || 99;
            const statusB = statusOrder[jobs[b].status] || 99;
            // Add secondary sort by time if timestamps existed
            return statusA - statusB;
        });

        for (const jobId of sortedJobIds) {
            const job = jobs[jobId];
            const jobHtml = formatJobStatus(jobId, job);

            if (job.status === 'completed' || job.status === 'failed') {
                finishedHtml += jobHtml;
                finishedCount++;
            } else { // 'queued' or 'running'
                activeHtml += jobHtml;
                activeCount++;
            }
        }

        activeJobsList.innerHTML = activeCount > 0 ? activeHtml : '<p class="text-muted p-2">No active jobs.</p>';
        finishedJobsList.innerHTML = finishedCount > 0 ? finishedHtml : '<p class="text-muted p-2">No finished jobs.</p>';
    }

    function fetchJobStatus() {
        fetch('/api/jobs') // Use the new API endpoint
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                updateJobQueue(data);
            })
            .catch(error => {
                console.error('Error fetching job status:', error);
                // Update error message display for new structure
                if (activeJobsList) activeJobsList.innerHTML = '<p class="text-danger p-2">Error loading job status.</p>';
                if (finishedJobsList) finishedJobsList.innerHTML = ''; // Clear finished jobs on error?
            });
    }

    // --- Initial Load and Polling Setup ---
    fetchJobStatus(); // Initial fetch
    setInterval(fetchJobStatus, 5000); // Poll every 5 seconds

});
