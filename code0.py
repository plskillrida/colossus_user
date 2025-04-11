import cv2
import os
import numpy as np

def load_gesture(letter):
    """
    Given a letter, build the folder path and try to load the gesture image (.jpg).
    """
    folder = letter.upper()
    image_path = os.path.join(os.getcwd(), 'mapping', folder, 'image.jpg')
    if os.path.exists(image_path):
        gesture_img = cv2.imread(image_path)
        if gesture_img is None:
            print(f"Failed to load image from {image_path}")
        return gesture_img
    else:
        print(f"Image not found for letter '{letter}' at path: {image_path}")
        return None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    selected_gesture = None
    selected_letter = None

    print("Press a key (B, C, D, ..., Y) to show the gesture image.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if key == ord('q'):
                break
            elif chr(key).upper() in [chr(i) for i in range(ord('A'), ord('Z')+1)] or chr(key).lower() == 'h':
                letter = chr(key)
                selected_letter = letter.upper()
                print(f"Selected: {selected_letter}")
                gesture = load_gesture(selected_letter)
                if gesture is not None:
                    gesture = cv2.resize(gesture, (frame.shape[1], frame.shape[0]))
                    selected_gesture = gesture
                else:
                    selected_gesture = None

        overlay_frame = frame.copy()
        if selected_letter:
            cv2.putText(overlay_frame, f"Letter: {selected_letter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if selected_gesture is not None:
            combined_image = np.hstack((overlay_frame, selected_gesture))
        else:
            combined_image = overlay_frame

        cv2.imshow("Real-time Sign Display", combined_image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
