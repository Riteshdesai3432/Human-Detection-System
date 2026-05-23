from fer import FER
import cv2

# Initialize emotion detector
detector = FER(mtcnn=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect emotions
    results = detector.detect_emotions(frame)

    # Loop through detected faces
    for result in results:

        # Face coordinates
        x, y, w, h = result["box"]

        # Get emotion with highest confidence
        emotions = result["emotions"]

        emotion_name = max(
            emotions,
            key=emotions.get
        )

        confidence = emotions[emotion_name]

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion text
        text = f"{emotion_name} ({confidence:.2f})"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Title
    cv2.putText(
        frame,
        "Emotion Detection System",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Show webcam
    cv2.imshow(
        "Emotion Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()

cv2.destroyAllWindows()

