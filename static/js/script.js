document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const audioFileInput = document.getElementById('audioFile');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const recordedAudio = document.getElementById('recordedAudio');
    const analyzeRecordingBtn = document.getElementById('analyzeRecording');
    const loadingIndicator = document.getElementById('loading');
    const resultsSection = document.getElementById('results');
    const initialState = document.getElementById('initial-state');
    const analyzedAudio = document.getElementById('analyzedAudio');
    const detectedEmotion = document.getElementById('detectedEmotion');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const emotionIcon = document.getElementById('emotionIcon');
    const emotionChart = document.getElementById('emotionChart');
    const waveformChart = document.getElementById('waveformChart');
    const spectrogramChart = document.getElementById('spectrogramChart');
    
    // Recording variables
    let mediaRecorder;
    let audioChunks = [];
    let audioBlob;
    
    // Event Listeners
    uploadForm.addEventListener('submit', handleUpload);
    recordButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);
    analyzeRecordingBtn.addEventListener('click', analyzeRecording);
    
    // Handle file upload
    function handleUpload(e) {
        e.preventDefault();
        
        const file = audioFileInput.files[0];
        if (!file) {
            alert('Please select an audio file');
            return;
        }
        
        // Show loading indicator
        showLoading();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send to server
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                hideLoading();
                return;
            }
            
            // Display results
            displayResults(data, file);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis');
            hideLoading();
        });
    }
    
    // Start audio recording
    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.addEventListener('dataavailable', event => {
                    audioChunks.push(event.data);
                });
                
                mediaRecorder.addEventListener('stop', () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    recordedAudio.src = audioUrl;
                    recordedAudio.classList.remove('d-none');
                    analyzeRecordingBtn.classList.remove('d-none');
                });
                
                // Start recording
                mediaRecorder.start();
                
                // Update UI
                recordButton.disabled = true;
                stopButton.disabled = false;
                recordButton.innerHTML = '<i class="fas fa-microphone"></i> Recording...';
                recordButton.classList.remove('btn-danger');
                recordButton.classList.add('btn-warning');
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                alert('Could not access microphone. Please check permissions.');
            });
    }
    
    // Stop audio recording
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            
            // Stop all tracks in the stream
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            // Update UI
            recordButton.disabled = false;
            stopButton.disabled = true;
            recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            recordButton.classList.remove('btn-warning');
            recordButton.classList.add('btn-danger');
        }
    }
    
    // Analyze recorded audio
    function analyzeRecording() {
        if (!audioBlob) {
            alert('No recording available');
            return;
        }
        
        // Show loading indicator
        showLoading();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', new File([audioBlob], 'recording.wav', { type: 'audio/wav' }));
        
        // Send to server
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                hideLoading();
                return;
            }
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis');
            hideLoading();
        });
    }
    
    // Display analysis results
    function displayResults(data, originalFile) {
        // Hide loading and initial state
        hideLoading();
        initialState.classList.add('d-none');
        
        // Show results section
        resultsSection.classList.remove('d-none');
        
        // Set audio source
        if (data.audio_path) {
            analyzedAudio.src = data.audio_path;
        } else if (originalFile) {
            analyzedAudio.src = URL.createObjectURL(originalFile);
        }
        
        // Update emotion and confidence
        detectedEmotion.textContent = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
        confidenceBar.style.width = `${data.confidence}%`;
        confidenceText.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
        
        // Update emotion icon
        updateEmotionIcon(data.emotion);
        
        // Update charts
        if (data.emotion_plot) {
            emotionChart.src = `data:image/png;base64,${data.emotion_plot}`;
        }
        
        if (data.waveform_plot) {
            waveformChart.src = `data:image/png;base64,${data.waveform_plot}`;
        }
        
        if (data.spectrogram_plot) {
            spectrogramChart.src = `data:image/png;base64,${data.spectrogram_plot}`;
        }
    }
    
    // Update emotion icon based on detected emotion
    function updateEmotionIcon(emotion) {
        emotionIcon.className = ''; // Clear existing classes
        
        // Add appropriate icon based on emotion
        switch(emotion) {
            case 'angry':
                emotionIcon.className = 'fas fa-face-angry emotion-icon';
                emotionIcon.style.color = '#FF5252';
                break;
            case 'calm':
                emotionIcon.className = 'fas fa-face-smile emotion-icon';
                emotionIcon.style.color = '#4CAF50';
                break;
            case 'disgust':
                emotionIcon.className = 'fas fa-face-dizzy emotion-icon';
                emotionIcon.style.color = '#9C27B0';
                break;
            case 'fearful':
                emotionIcon.className = 'fas fa-face-fearful emotion-icon';
                emotionIcon.style.color = '#2196F3';
                break;
            case 'happy':
                emotionIcon.className = 'fas fa-face-laugh-beam emotion-icon';
                emotionIcon.style.color = '#FFEB3B';
                break;
            case 'neutral':
                emotionIcon.className = 'fas fa-face-meh emotion-icon';
                emotionIcon.style.color = '#607D8B';
                break;
            case 'sad':
                emotionIcon.className = 'fas fa-face-sad-tear emotion-icon';
                emotionIcon.style.color = '#3F51B5';
                break;
            case 'surprised':
                emotionIcon.className = 'fas fa-face-surprise emotion-icon';
                emotionIcon.style.color = '#FF9800';
                break;
            default:
                emotionIcon.className = 'fas fa-face-smile emotion-icon';
                emotionIcon.style.color = '#2196F3';
        }
    }
    
    // Show loading indicator
    function showLoading() {
        loadingIndicator.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        initialState.classList.add('d-none');
    }
    
    // Hide loading indicator
    function hideLoading() {
        loadingIndicator.classList.add('d-none');
    }
});
