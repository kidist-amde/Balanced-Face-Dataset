import numpy as np
import cv2
import dlib
import tensorflow as tf


def main():
    
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    model = tf.keras.models.load_model("trained-models/eye-state/eye-state-detection-model.h5")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        eye= ""
        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2  = face.left(), face.top(), face.right(), face.bottom()
            x1 = max(0, x1 - 20)
            x2 = min(frame.shape[1], x2 + 20)
            y1 = max(0, y1 - 20)
            y2 = min(frame.shape[0], y2 + 20)
            face_image = frame[y1:y2, x1:x2, :]
            
            face_image = cv2.resize(face_image, (224, 224))
            face_image = np.expand_dims(face_image, axis=0)
            face_image = face_image.astype(np.float32)/255.0
            pred = model.predict(face_image)
            if pred[0][0] > 0.5:
                eye = "open"
            else:
                eye = "closed"
                
            cv2.putText(frame, "Eye: {}".format(eye), (x1 - 10, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('eye:state',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    