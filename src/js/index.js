const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const uploadButton = document.getElementById('upload');
const resultDiv = document.getElementById('result');

captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    video.style.display = 'none';
    canvas.style.display = 'block';
    uploadButton.style.display = 'block';
    
});

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
    });

uploadButton.addEventListener('click', () => {
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'photo.png');

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(result => {
            resultDiv.innerHTML = result;
        })
        .catch(error => {
            console.error('Error uploading image:', error);
        });
    }, 'image/png');
});
