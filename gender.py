from deepface import DeepFace
import cv2

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    try:

        # Analyze gender
        result = DeepFace.analyze(
            frame,
            actions=['gender'],
            enforce_detection=False
        )

        # Get dominant gender
        gender = result[0]['dominant_gender']

        # Display gender
        cv2.putText(
            frame,
            f"Gender: {gender}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print(e)

    # Show webcam
    cv2.imshow("Gender Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
