from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
import speech_recognition as sr

app = Flask(__name__)
camera = cv2.VideoCapture(0)
selected_letter = None

def load_gesture(letter):
    folder = letter.upper()
    image_path = os.path.join('static', 'mapping', folder, 'image.jpg')
    if os.path.exists(image_path):
        gesture_img = cv2.imread(image_path)
        return cv2.resize(gesture_img, (640, 480))
    return None

def generate_frames():
    global selected_letter
    while True:
        success, frame = camera.read()
        if not success:
            break

        overlay_frame = frame.copy()

        gesture = None
        if selected_letter:
            cv2.putText(overlay_frame, f"Letter: {selected_letter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture = load_gesture(selected_letter)

        if gesture is not None:
            gesture = cv2.resize(gesture, (640, 480))
            combined = np.hstack((overlay_frame, gesture))
        else:
            combined = overlay_frame

        _, buffer = cv2.imencode('.jpg', combined)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    global selected_letter
    if request.method == 'POST':
        selected_letter = request.form.get('letter').upper()
    return render_template('input.html', selected_letter=selected_letter)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/voice', methods=['GET', 'POST'])
def voice_input():
    if request.method == 'POST':
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Capture audio from the microphone
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Please say a letter...")
            audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            spoken_letter = recognizer.recognize_google(audio)
            print(f"Recognized letter: {spoken_letter.upper()}")
            
            # Process the recognized letter and return the corresponding image URL
            letter_image_url = f"/static/letters/{spoken_letter.upper()}.png"  # Ensure your letter images are stored in the static folder

            return render_template('voice.html', letter=spoken_letter.upper(), letter_image_url=letter_image_url)

        except sr.UnknownValueError:
            return render_template('voice.html', error="Could not understand the audio.")
        except sr.RequestError:
            return render_template('voice.html', error="Could not request results from Google Speech Recognition service.")
    
    return render_template('voice.html')


if __name__ == '__main__':
    app.run(debug=True)
