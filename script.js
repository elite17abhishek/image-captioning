document.getElementById('upload-image').addEventListener('submit', async function(event) {
    event.preventDefault();
    let imageInput = document.getElementById('image');
    let image = imageInput.files[0];
    
    let formData = new FormData();
    formData.append('image', image);
    
    try {
        let response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        let result = await response.json();
        
        let resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<p>Class: ${result.class}</p>`;
    } catch (error) {
        console.error('Error:', error);
    }
});
