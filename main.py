import cv2
import numpy as np

# Load YOLO Tiny model
net = cv2.dnn.readNet(
    "yolov3-tiny.weights",
    "yolov3-tiny.cfg"
)

# Load COCO names
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()

output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers()
]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width, channels = frame.shape

    # Create blob
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (320, 320),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    # Human detection
    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

            # Detect only PERSON
            if class_id == 0 and confidence > 0.3:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Remove duplicate boxes
    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        0.3,
        0.4
    )

    person_count = 0

    # Draw human boxes
    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            person_count += 1

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Person",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        5
    )

    # Draw face boxes
    for (fx, fy, fw, fh) in faces:

        cv2.rectangle(
            frame,
            (fx, fy),
            (fx + fw, fy + fh),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (fx, fy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # Display people count
    cv2.putText(
        frame,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Show output
    cv2.imshow(
        "Human Detection System",
        frame
    )

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
