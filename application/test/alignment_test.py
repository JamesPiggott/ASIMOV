import cv2
from application.core.API import API
from application.core.api.detection.Alignment import alignment_procedure


def main():
    api = API()

    # Load the slightly misaligned image of the actress
    frame = cv2.imread('sample_images/scarlett_johansson.jpg')
    if frame is None:
        raise FileNotFoundError("Image file not found")

    result = api.detect_faces(frame, True)
    if not result.faces:
        raise ValueError("No faces detected in the image")

    # Attempt to align the face
    aligned = alignment_procedure(result.faces[0])

    while True:
        cv2.imshow("Aligned Face", aligned)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
