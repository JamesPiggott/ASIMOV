import cv2
from application.core.API import API
from application.core.api.detection.Alignment import alignment_procedure


def save_face_crop(image_path, output_path, api):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    result = api.detect_faces(frame, True)
    if not result.faces:
        raise ValueError(f"No faces detected in the image at {image_path}")

    crop = result.faces[0].get_crop()
    cv2.imwrite(output_path, crop)
    return result.faces[0]


def compare_and_print(api, description, crop1, crop2):
    distance = api.compare_faces(crop1, crop2)
    print(f"{description} : {distance}")


def main():
    api = API()

    # Save face crops
    face1 = save_face_crop('sample_images/scarlett_johansson.jpg', 'sample_images/scarlett_crop.jpg', api)
    face2 = save_face_crop('sample_images/scarlett_johansson2.jpg', 'sample_images/scarlett_crop2.jpg', api)
    face3 = save_face_crop('sample_images/scarlett22.jpg', 'sample_images/scarlett22_crop.jpg', api)
    face4 = save_face_crop('sample_images/alicia_vikander.jpg', 'sample_images/alicia_crop.jpg', api)
    face5 = save_face_crop('sample_images/alicia_lookalike.JPG', 'sample_images/alicia_lookalike_crop.jpg',
                           api)
    face6 = save_face_crop('sample_images/michael_fassbender.jpg', 'sample_images/michael_crop.jpg', api)

    # Compare the same crops
    compare_and_print(api, "Scarlett 2 times the same", face1.get_crop(), face1.get_crop())

    # Compare two different crops of the same person
    compare_and_print(api, "Scarlett 2 times different", face1.get_crop(), face2.get_crop())
    aligned1 = alignment_procedure(face1)
    aligned2 = alignment_procedure(face2)
    compare_and_print(api, "Scarlett 2 times different aligned", aligned1, aligned2)

    # Compare different crops of Scarlett Johansson
    compare_and_print(api, "Scarlett 2 times different", face1.get_crop(), face3.get_crop())
    aligned1 = alignment_procedure(face1)
    aligned2 = alignment_procedure(face3)
    compare_and_print(api, "Scarlett 2 times different aligned", aligned1, aligned2)

    # Compare crops of Scarlett Johansson and Alicia Vikander
    compare_and_print(api, "Scarlett + Alicia", face1.get_crop(), face4.get_crop())
    aligned1 = alignment_procedure(face1)
    aligned2 = alignment_procedure(face4)
    compare_and_print(api, "Scarlett + Alicia aligned", aligned1, aligned2)

    # Compare Alicia Vikander with Michael Fassbender
    compare_and_print(api, "Michael + Alicia", face4.get_crop(), face6.get_crop())
    aligned1 = alignment_procedure(face4)
    aligned2 = alignment_procedure(face6)
    compare_and_print(api, "Michael + Alicia aligned", aligned1, aligned2)

    # Compare Alicia Vikander with her lookalike
    compare_and_print(api, "Alicia + lookalike", face4.get_crop(), face5.get_crop())
    aligned1 = alignment_procedure(face4)
    aligned2 = alignment_procedure(face5)
    compare_and_print(api, "Alicia + lookalike aligned", aligned1, aligned2)


if __name__ == "__main__":
    main()
