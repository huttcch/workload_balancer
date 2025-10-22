document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const excelFile = document.getElementById('excelFile');
    const numPeople = document.getElementById('numPeople');
    const submitBtn = document.getElementById('submitBtn');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');
    const resultMessage = document.getElementById('resultMessage');
    const finalVariance = document.getElementById('finalVariance');
    const downloadLink = document.getElementById('downloadLink');

    uploadForm.addEventListener('submit', async function(event) {
        event.preventDefault(); // Prevent default form submission

        // Hide previous results/errors
        resultsDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');
        loadingDiv.classList.remove('hidden');
        submitBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', excelFile.files[0]);
        formData.append('num_people', numPeople.value);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                loadingDiv.classList.add('hidden');
                resultsDiv.classList.remove('hidden');
                resultMessage.textContent = data.message;
                finalVariance.textContent = data.variance.toFixed(2);
                downloadLink.href = data.download_url; // Set the download URL
            } else {
                loadingDiv.classList.add('hidden');
                errorDiv.classList.remove('hidden');
                errorMessage.textContent = data.error || 'An unknown error occurred.';
            }
        } catch (error) {
            loadingDiv.classList.add('hidden');
            errorDiv.classList.remove('hidden');
            errorMessage.textContent = 'Network error or server unavailable: ' + error.message;
            console.error('Fetch error:', error);
        } finally {
            submitBtn.disabled = false;
        }
    });
});