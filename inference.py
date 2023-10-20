import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load your Keras model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Define a dictionary that maps predictions to emoji images
emoji_map = {
    "happy": cv2.imread("happy_emoji.png"),
    "happy2": cv2.imread("happy_emoji.png"),
    "happy3": cv2.imread("happy_emoji.png"),
    "sad": cv2.imread("sad_emoji.png"),
    "sad2": cv2.imread("sad_emoji.png"),
    "nice": cv2.imread("nice_emoji.png"),
    "nice2": cv2.imread("nice_emoji.png"),
    "surprised": cv2.imread("surprised_emoji.png"),
    "angry": cv2.imread("angry_emoji.png"),
    "angry2": cv2.imread("angry_emoji.png"),
    "shy": cv2.imread("shy_emoji.png"),
    # Add more mappings for your predictions and corresponding emoji images
}

# Initialize Mediapipe components
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    lst = []

    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]        
        emoji_image = emoji_map.get(pred, None)
        

        if emoji_image is not None:
            emoji_image = cv2.resize(emoji_image, (100, 100))
            x_offset, y_offset = 50, 150
            frm[y_offset:y_offset+emoji_image.shape[0], x_offset:x_offset+emoji_image.shape[1]] = emoji_image
        

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
