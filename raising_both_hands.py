import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict

# Definition of Keypoints type
# xy: ndarray or Tensor with shape (num_keypoints, 2)
# conf: ndarray or Tensor with shape (num_keypoints,)
Keypoints = Dict[str, np.ndarray]  # Could also use torch.Tensor depending on the implementation

# Load the YOLO model
def load_model(model_path):
    """
    Loads a YOLO model and transfers it to the MPS device for acceleration.
    
    Parameters
    ----------
    model_path : str
        Path to the YOLO model file.

    Returns
    -------
    model : YOLO
        Loaded YOLO model instance transferred to MPS.
    """
    # Load YOLO models and transfer them to MPS devices
    # MPS: Use Apple's Metal Performance Shaders to speed up models
    # Delete .to(‘mps’) when mps is not available
    return YOLO(model_path).to("mps")

# Define keypoints names
KEYPOINTS_NAMES = [
    "nose", "eye(L)", "eye(R)", "ear(L)", "ear(R)",
    "shoulder(L)", "shoulder(R)", "elbow(L)", "elbow(R)",
    "wrist(L)", "wrist(R)", "hip(L)", "hip(R)", "knee(L)",
    "knee(R)", "ankle(L)", "ankle(R)",
]

# Define connections between landmarks (joints)
JOINTS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (11, 12), (11, 13), (12, 14),
    (13, 15), (14, 16), (5, 11), (6, 12)
]

# Inference process
def process_frame(frame: np.ndarray, model: YOLO):
    """
    Processes a video frame using the YOLO model and detects keypoints.

    Parameters
    ----------
    frame : numpy.ndarray
        The video frame to process.
    model : YOLO
        Instance of YOLO model used for keypoint detection.

    Returns
    -------
    results : Results
        YOLO model results containing detected keypoints.
    """
    results = model(frame)
    return results[0]

# Draw keypoints
def draw_keypoints(frame: np.ndarray, keypoints_list: list) -> np.ndarray:
    """
    Draws keypoints and connections on the frame.

    Parameters
    ----------
    frame : numpy.ndarray
        The image frame where keypoints will be drawn.
    keypoints_list : list of Keypoints
        List of keypoints to be drawn.

    Returns
    -------
    annotated_frame : numpy.ndarray
        The frame with drawn keypoints and connections.
    """
    annotated_frame = frame.copy()
    
    for keypoints in keypoints_list:
        if keypoints.conf is None or keypoints.xy is None:
            continue

        confs = keypoints.conf[0].tolist()  # Inference results: closer to 1, higher confidence
        xys = keypoints.xy[0].tolist()  # Coordinates

        # Draw connections between keypoints
        for (start, end) in JOINTS:
            if confs[start] >= 0.7 and confs[end] >= 0.7:
                start_point = (int(xys[start][0]), int(xys[start][1]))
                end_point = (int(xys[end][0]), int(xys[end][1]))
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)

        # Draw keypoints
        for index, (xy, conf) in enumerate(zip(xys, confs)):
            if conf < 0.7:
                continue

            x, y = int(xy[0]), int(xy[1])
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + 3, y + 3),
                (255, 0, 255),
                cv2.FILLED,
                cv2.LINE_AA,
            )

    return annotated_frame

# Draw bounding boxes
def draw_bounding_boxes(frame: np.ndarray, keypoints_list: list) -> tuple[np.ndarray, int]:
    """
    Draws bounding boxes around detected keypoints and counts people with both hands raised.

    Parameters
    ----------
    frame : numpy.ndarray
        The image frame where bounding boxes will be drawn.
    keypoints_list : list of Keypoints
        List of detected keypoints for bounding box drawing.

    Returns
    -------
    annotated_frame : numpy.ndarray
        Frame with drawn bounding boxes.
    both_hands_people : int
        The number of people detected with both hands raised.
    """
    annotated_frame = frame.copy()

    both_hands_people = 0
    for keypoints in keypoints_list:
        if keypoints.conf is None or keypoints.xy is None:
            continue

        xys = keypoints.xy[0].tolist()  # Coordinates
        confs = keypoints.conf[0].tolist()  # Inference results: closer to 1, higher confidence

        # Calculate bounding box coordinates
        x_coords = [int(xy[0]) for xy, conf in zip(xys, confs) if conf >= 0.7]
        y_coords = [int(xy[1]) for xy, conf in zip(xys, confs) if conf >= 0.7]

        if not x_coords or not y_coords:
            continue

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        if is_raising_both_hands(keypoints):
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            both_hands_people += 1
        else:
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return annotated_frame, both_hands_people

# Check if both hands are raised
def is_raising_both_hands(keypoints: Keypoints) -> bool:
    """
    Checks if a person is raising both hands above their head.

    Parameters
    ----------
    keypoints : Keypoints
        Dictionary containing detected keypoints with keys 'xy' (coordinates) and 'conf' (confidence).

    Returns
    -------
    bool
        True if both hands are raised above the head, False otherwise.
    """
    if len(keypoints.xy[0]) < 17:
        return False

    wrist_left_idx = 9
    wrist_right_idx = 10
    nose_idx = 0

    # Check if the keypoints are present
    if (keypoints.conf[0][wrist_left_idx] is None or
        keypoints.conf[0][wrist_right_idx] is None or
        keypoints.conf[0][nose_idx] is None):
        return False

    # Get the coordinates of the wrists and nose
    wrists_left = keypoints.xy[0][wrist_left_idx]
    wrists_right = keypoints.xy[0][wrist_right_idx]
    nose = keypoints.xy[0][nose_idx]

    if((wrists_left[0] == 0 and wrists_left[1] == 0) or
       (wrists_right[0] == 0 and wrists_right[1] == 0) or
       (nose[0] == 0 and nose[1] == 0)):
        return False

    # Get the confidence scores
    conf_left = keypoints.conf[0][wrist_left_idx]
    conf_right = keypoints.conf[0][wrist_right_idx]
    conf_nose = keypoints.conf[0][nose_idx]

    # Check if both wrists are above the nose
    if (wrists_left[1] < nose[1] and wrists_right[1] < nose[1]):
        return True

    return False

# Filter keypoints based on confidence
def filter_keypoints(keypoints_list: list, threshold: float = 0.8) -> list:
    """
    Filters keypoints based on confidence threshold.

    Parameters
    ----------
    keypoints_list : list of Keypoints
        List of detected keypoints to filter.
    threshold : float, optional
        Confidence threshold for filtering keypoints.

    Returns
    -------
    filtered_keypoints_list : list of Keypoints
        List of keypoints with confidence above the threshold.
    """
    return [
        kp for kp in keypoints_list
        if kp.conf is not None and max(kp.conf[0]) >= threshold
    ]

# Sort keypoints by confidence
def sort_keypoints_by_confidence(keypoints_list: list) -> list:
    """
    Sorts keypoints in descending order of confidence scores.

    Parameters
    ----------
    keypoints_list : list of Keypoints
        List of detected keypoints to sort.

    Returns
    -------
    sorted_keypoints_list : list of Keypoints
        Keypoints sorted by highest confidence scores.
    """
    return sorted(
        keypoints_list,
        key=lambda kp: max(kp.conf[0]) if kp.conf is not None else 0,
        reverse=True
    )

# メイン処理
def main(video_path: str, model_path: str) -> None:
    """
    Main function to process video frames, detect keypoints, draw annotations, and display the results.

    Parameters
    ----------
    video_path : str
        Path to the video file to process.
    model_path : str
        Path to the YOLO model file.

    Returns
    -------
    None
    """
    model = load_model(model_path)
    capture = cv2.VideoCapture(video_path)

    current_both_hands_people = 0

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        # Process the frame to detect keypoints
        results = process_frame(frame, model)

        keypoints_list = results.keypoints

        # Filter keypoints based on confidence
        filtered_keypoints_list = filter_keypoints(keypoints_list, threshold=0.85)
        sorted_keypoints_list = sort_keypoints_by_confidence(filtered_keypoints_list)

        # Get the top 5 keypoints based on confidence
        top_5_keypoints_list = sorted_keypoints_list[:10]

        # Print the number of detected keypoints
        print(f"Number of keypoints: {len(top_5_keypoints_list)}")

        # Draw keypoints on the frame
        if top_5_keypoints_list:
            result_img = draw_keypoints(frame, top_5_keypoints_list)
        else:
            result_img = frame

        # Draw bounding boxes around detected keypoints
        if top_5_keypoints_list:
            result_img, both_hands_people = draw_bounding_boxes(result_img, top_5_keypoints_list)
        else:
            both_hands_people = 0

        if both_hands_people > current_both_hands_people:
            print("Both hands raised!")

        print(f"Both hands people: {both_hands_people}")
        current_both_hands_people = both_hands_people

        cv2.putText(result_img, f"Both hands people: {both_hands_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the processed frame
        cv2.imshow("YOLOv8 keypoints and bounding boxes", result_img)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    capture.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = "path/to/video.mp4"
    model_path = "path/to/yolov8_pose_models.pt"
    main(video_path, model_path)
