from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

def recognize_faces():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('my_face_recognizer_model.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (720, 720), interpolation=cv2.INTER_CUBIC)
            
            # Reconocimiento facial
            label, confidence = face_recognizer.predict(face)
            if confidence < 25:
                name = 'Aylin <3'
            else:
                name = 'Desconocido'
            
            # Mostrar nombre y confianza
            text = f'{name}: {confidence:.2f}'
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
