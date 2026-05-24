from fer import FER
import cv2

# Initialize emotion detector
detector = FER(mtcnn=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect emotions
    results = detector.detect_emotions(frame)

    for result in results:

        x, y, w, h = result["box"]

        emotions = result["emotions"]

        # Get highest emotion
        emotion = max(
            emotions,
            key=emotions.get
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Show webcam
    cv2.imshow(
        "Emotion Detection",
        frame
    )

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

