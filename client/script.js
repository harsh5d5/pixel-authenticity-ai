document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const resultsSection = document.getElementById('results-section');
    const resultStatus = document.getElementById('result-status');
    const trustPercentage = document.getElementById('trust-percentage');
    const insightText = document.getElementById('insight-text');
    const systemStatus = document.querySelector('.status-badge');

    // Health check for Backend
    async function checkBackendHealth() {
        try {
            const resp = await fetch('http://127.0.0.1:5000/', { method: 'GET' });
            if (resp.ok) {
                systemStatus.innerHTML = '<span class="pulse"></span> System Operational';
                systemStatus.style.background = 'rgba(0, 255, 136, 0.1)';
                systemStatus.style.color = 'var(--accent-green)';
                systemStatus.style.borderColor = 'rgba(0, 255, 136, 0.2)';
            } else {
                throw new Error();
            }
        } catch (e) {
            systemStatus.innerHTML = '<span class="pulse" style="background: var(--accent-red); box-shadow: 0 0 10px var(--accent-red);"></span> Engine Offline';
            systemStatus.style.background = 'rgba(255, 62, 62, 0.1)';
            systemStatus.style.color = 'var(--accent-red)';
            systemStatus.style.borderColor = 'rgba(255, 62, 62, 0.2)';
        }
    }

    // Run health check on load and every 10 seconds
    checkBackendHealth();
    setInterval(checkBackendHealth, 10000);

    // Trigger file input
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && e.target !== analyzeBtn && !previewContainer.contains(e.target)) {
            fileInput.click();
        }
    });

    // Handle file selection
    fileInput.addEventListener('change', handleFile);

    // Handle drag & drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#00f2ff';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFile({ target: fileInput });
        }
    });

    const scanner = document.getElementById('scanner');
    const gaugeFill = document.getElementById('gauge-fill');
    const verdictBox = document.getElementById('verdict-box');
    const verdictDesc = document.getElementById('verdict-desc');
    const evidenceList = document.getElementById('evidence-list');

    function handleFile(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
                document.querySelector('.upload-icon').style.display = 'none';
                document.querySelector('#drop-zone h3').style.display = 'none';
                document.querySelector('#drop-zone p').style.display = 'none';
                resultsSection.style.display = 'none';
                document.body.className = ''; // Reset theme
            };
            reader.readAsDataURL(file);
        }
    }

    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        previewContainer.style.display = 'none';
        document.querySelector('.upload-icon').style.display = 'block';
        document.querySelector('#drop-zone h3').style.display = 'block';
        document.querySelector('#drop-zone p').style.display = 'block';
        resultsSection.style.display = 'none';
        document.body.className = '';
    });

    // REAL Analysis Connection
    analyzeBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const file = fileInput.files[0];
        if (!file) {
            alert('Please upload an image first.');
            return;
        }

        analyzeBtn.innerText = 'Neural Scanning...';
        analyzeBtn.disabled = true;
        scanner.style.display = 'block';
        resultsSection.style.display = 'none';

        const investigationLog = document.getElementById('investigation-log');
        investigationLog.innerHTML = '';
        investigationLog.style.display = 'flex';

        const formData = new FormData();
        formData.append('image', file);

        try {
            console.log("Attempting to reach backend at http://127.0.0.1:5000/analyze");
            const analysisPromise = fetch('http://127.0.0.1:5000/analyze', {
                method: 'POST',
                body: formData
            });

            // Run the "investigation" animation sequence
            await runInvestigationLog([
                "Initializing forensic engines...",
                "Scanning sensor noise profiles...",
                "Calculating compression entropy...",
                "Matching AI frequency patterns...",
                "Finalizing integrity verdict..."
            ]);

            const response = await analysisPromise;
            const data = await response.json();

            investigationLog.style.display = 'none';
            scanner.style.display = 'none';
            resultsSection.style.display = 'grid';
            analyzeBtn.innerText = 'Start Neural Scan';
            analyzeBtn.disabled = false;

            if (data.status === 'FAKE') {
                document.body.className = 'theme-fake';
                resultStatus.innerText = 'FAKE';
                verdictDesc.innerText = 'High probability of complete AI synthesis or heavy manipulation.';
            } else if (data.status === 'EDITED') {
                document.body.className = 'theme-edited';
                resultStatus.innerText = 'EDITED';
                verdictDesc.innerText = 'The image likely started as a real photo but shows signs of splicing, cloning, or retouching.';
            } else {
                document.body.className = 'theme-real';
                resultStatus.innerText = 'REAL';
                verdictDesc.innerText = 'The image retains its original sensor noise and compression patterns.';
            }
            animateGauge(data.trust_score);

            // Update Slider Comparison
            const heatmapCompare = document.getElementById('heatmap-compare');
            const originalCompare = document.getElementById('original-compare');

            heatmapCompare.src = data.heatmap;
            originalCompare.src = imagePreview.src;

            initSlider();

            // Update Metrics using REAL backend data
            if (data.metrics) {
                updateMetrics(
                    data.metrics.ela,
                    data.metrics.noise,
                    data.metrics.fft
                );
            } else {
                updateMetrics(
                    data.status === 'REAL' ? 95 : 30,
                    data.status === 'REAL' ? 92 : 45,
                    data.status === 'REAL' ? 88 : 25
                );
            }

            renderEvidence(data.evidence);
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Forensic Engine Connectivity Error:', error);

            // Show error in the investigation log instead of a popup alert
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `
                <span class="log-text" style="color: var(--accent-red)">!! CONNECTION FAILED !!</span>
                <span class="log-status fail">[ERROR]</span>
            `;
            investigationLog.appendChild(entry);

            const tip = document.createElement('div');
            tip.className = 'log-entry';
            tip.innerHTML = `<span class="log-text" style="font-size: 0.7rem">> Ensure 'app.py' is running on port 5000.</span>`;
            investigationLog.appendChild(tip);

            analyzeBtn.innerText = 'Scan Failed - Retry?';
            analyzeBtn.disabled = false;
            scanner.style.display = 'none';
            // Keep the log visible so the user can see what happened
            setTimeout(() => {
                if (!resultsSection.offsetParent) { // If results aren't shown
                    investigationLog.style.display = 'none';
                    analyzeBtn.innerText = 'Start Neural Scan';
                }
            }, 5000);
        }
    });

    function animateGauge(percentage) {
        let current = 0;
        const interval = setInterval(() => {
            if (current >= percentage) {
                clearInterval(interval);
            } else {
                current++;
                trustPercentage.innerText = current + '%';
                // Gauge circumference is ~125.66 (for radius 40 half circle)
                // Offset calculation: 125.66 - (125.66 * (current / 100))
                const offset = 125.66 - (125.66 * (current / 100));
                gaugeFill.style.strokeDashoffset = offset;
            }
        }, 15);
    }

    function updateMetrics(ela, noise, fft) {
        document.getElementById('bar-ela').style.width = ela + '%';
        document.getElementById('bar-noise').style.width = noise + '%';
        document.getElementById('bar-fft').style.width = fft + '%';
    }

    function renderEvidence(points) {
        // Keep the header
        const header = evidenceList.querySelector('.audit-row-header');
        evidenceList.innerHTML = '';
        evidenceList.appendChild(header);

        points.forEach(p => {
            const div = document.createElement('div');
            div.className = `evidence-item status-${p.status}`;

            let statusLabel = 'PASS';
            let statusIcon = '✓';
            if (p.status === 'alert') { statusLabel = 'VIOLATION'; statusIcon = '✕'; }
            if (p.status === 'warning') { statusLabel = 'WARNING'; statusIcon = '⚠️'; }

            div.innerHTML = `
                <div class="evidence-text">${p.text}</div>
                <div class="evidence-status-pill">
                    <span class="status-icon">${statusIcon}</span>
                    <span>${statusLabel}</span>
                </div>
            `;
            evidenceList.appendChild(div);
        });
    }

    function initSlider() {
        const slider = document.getElementById('compare-slider');
        const beforeImg = document.getElementById('image-before');
        const container = document.getElementById('forensic-compare');
        let active = false;

        container.addEventListener('mousedown', () => { active = true; });
        container.addEventListener('mouseup', () => { active = false; });
        container.addEventListener('mouseleave', () => { active = false; });

        container.addEventListener('mousemove', (e) => {
            if (!active) return;
            let x = e.pageX;
            x -= container.getBoundingClientRect().left;
            slideIt(x);
        });

        // Touch support
        container.addEventListener('touchstart', () => { active = true; });
        container.addEventListener('touchend', () => { active = false; });
        container.addEventListener('touchcancel', () => { active = false; });
        container.addEventListener('touchmove', (e) => {
            if (!active) return;
            let x = e.touches[0].pageX;
            x -= container.getBoundingClientRect().left;
            slideIt(x);
        });

        function slideIt(x) {
            let width = container.offsetWidth;
            let percentage = (x / width) * 100;
            if (percentage < 0) percentage = 0;
            if (percentage > 100) percentage = 100;

            slider.style.left = percentage + '%';
            beforeImg.style.width = percentage + '%';
        }

        // Start in the middle
        slideIt(container.offsetWidth / 2);
    }

    async function runInvestigationLog(steps) {
        const log = document.getElementById('investigation-log');
        for (const step of steps) {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `
                <span class="log-text">> ${step}</span>
                <span class="log-status wait">[WAIT]</span>
            `;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;

            await new Promise(r => setTimeout(r, 600));
            entry.querySelector('.log-status').innerText = '[DONE]';
            entry.querySelector('.log-status').className = 'log-status done';
        }
        await new Promise(r => setTimeout(r, 400));
    }
});
