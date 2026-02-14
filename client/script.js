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

    // Trigger file input
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !previewContainer.contains(e.target)) {
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
    analyzeBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('Please upload an image first.');
            return;
        }

        analyzeBtn.innerText = 'Neural Scanning...';
        analyzeBtn.disabled = true;
        scanner.style.display = 'block';
        resultsSection.style.display = 'none';

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

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

            // Update Heatmap
            const heatmapImg = document.getElementById('result-heatmap');
            heatmapImg.src = data.heatmap;
            heatmapImg.style.display = 'block';
            document.querySelector('.heatmap-overlay').style.display = 'none';

            // Update Metrics using REAL backend data
            if (data.metrics) {
                updateMetrics(
                    data.metrics.ela,
                    data.metrics.noise,
                    data.metrics.fft
                );
            } else {
                // Fallback for older API versions
                updateMetrics(
                    data.status === 'REAL' ? 95 : 30,
                    data.status === 'REAL' ? 92 : 45,
                    data.status === 'REAL' ? 88 : 25
                );
            }

            // Render Real Evidence Points
            renderEvidence(data.evidence);

            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert('Backend server not responding. Please make sure app.py is running.');
            analyzeBtn.innerText = 'Start Neural Scan';
            analyzeBtn.disabled = false;
            scanner.style.display = 'none';
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
        evidenceList.innerHTML = '';
        points.forEach(p => {
            const div = document.createElement('div');
            div.className = 'evidence-item';
            div.innerHTML = `
                <div class="evidence-status status-${p.status}"></div>
                <div class="evidence-text">${p.text}</div>
            `;
            evidenceList.appendChild(div);
        });
    }
});
