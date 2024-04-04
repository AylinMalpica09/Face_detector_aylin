const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const uploadButton = document.getElementById('upload');
const newPhotoButton = document.getElementById('new-photo');
const resultDiv = document.getElementById('result');

let stream;

captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    video.style.display = 'none';
    canvas.style.display = 'block';
    uploadButton.style.display = 'block';
    newPhotoButton.style.display = 'block'; // Mostrar el botón para tomar una nueva foto
});

newPhotoButton.addEventListener('click', () => {
    video.style.display = 'block';
    canvas.style.display = 'none';
    uploadButton.style.display = 'none';
    newPhotoButton.style.display = 'none'; // Ocultar el botón para tomar una nueva foto
});

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        stream = stream;
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
    });

captureButton.addEventListener('click', () => {
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
