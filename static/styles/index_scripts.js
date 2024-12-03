// Form validation and interaction
document.addEventListener("DOMContentLoaded", function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');
    
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        // Validate file input
        if (fileInput.files.length === 0) {
            resultDiv.textContent = "Please select a file.";
            resultDiv.style.color = "red";
        } else {
            resultDiv.textContent = "File uploaded successfully!";
            resultDiv.style.color = "green";
        }
    });
});

// Smooth scrolling for navbar links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
