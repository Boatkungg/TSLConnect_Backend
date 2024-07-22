import os

import cv2
import numpy as np
import pandas as pd
import torch
from mediapipe.python.solutions import holistic as mp_holistic

lips_outside = [
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
    61,
    185,
    40,
    39,
    37,
]
lips_inside = [
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
]
left_eye = [
    386,
    387,
    388,
    466,
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
]
right_eye = [
    159,
    160,
    161,
    246,
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
]
left_eyebrow = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
right_eyebrow = [107, 66, 105, 63, 70, 55, 65, 52, 53, 46]
face_outline = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]

only_face = []
only_face.extend(lips_inside)
only_face.extend(left_eye)
only_face.extend(right_eye)
only_face.extend(left_eyebrow)
only_face.extend(right_eyebrow)
only_face.extend(face_outline)

only_pose = [0, 8, 7, 11, 13, 15, 12, 14, 16, 23, 24]


def resize_and_pad(
    landmarks, original_width, original_height, target_width=854, target_height=480
):
    """
    Resize landmarks to fit the height of the target size, center them,
    and add padding to reach the target width.
    """
    # Calculate scaling factor based on height
    scale = target_height / original_height

    # Calculate new width after scaling
    new_width = int(original_width * scale)

    # Resize landmarks
    landmarks_resized = landmarks.copy()
    landmarks_resized[0::3] = landmarks_resized[0::3] * scale  # x coordinates
    landmarks_resized[1::3] = landmarks_resized[1::3] * scale  # y coordinates

    # Calculate padding
    pad_left = (target_width - new_width) // 2

    # Add padding to x coordinates
    landmarks_resized[0::3][landmarks_resized[0::3] != 0] += pad_left

    return landmarks_resized


def extract_keypoints(
    video_path, mp_holistic=mp_holistic, target_width=854, target_height=480
):
    name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_holistic.Holistic(
        static_image_mode=False, min_detection_confidence=0.5, model_complexity=2
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # print(results.face_landmarks)
            if results.pose_landmarks is None:
                print(name, "No pose landmarks")
            if results.face_landmarks is None:
                print(name, "No face landmarks")
            if results.left_hand_landmarks is None:
                print(name, "No left hand landmarks")
            if results.right_hand_landmarks is None:
                print(name, "No right hand landmarks")

            # Extract keypoints (customize based on your needs)
            pose = (
                np.array(
                    [[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]
                )[only_pose].flatten()
                if results.pose_landmarks
                else np.zeros(11 * 4)
            )
            face = (
                np.array(
                    [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                )[only_face].flatten()
                if results.face_landmarks
                else np.zeros(108 * 3)
            )
            lh = (
                np.array(
                    [
                        [res.x, res.y, res.z]
                        for res in results.left_hand_landmarks.landmark
                    ]
                ).flatten()
                if results.left_hand_landmarks
                else np.zeros(21 * 3)
            )
            rh = (
                np.array(
                    [
                        [res.x, res.y, res.z]
                        for res in results.right_hand_landmarks.landmark
                    ]
                ).flatten()
                if results.right_hand_landmarks
                else np.zeros(21 * 3)
            )

            frame_landmarks = np.concatenate([pose, face, lh, rh])

            # Denormalize landmarks
            frame_landmarks[0::3] *= original_width  # x coordinates
            frame_landmarks[1::3] *= original_height  # y coordinates

            frame_landmarks = resize_and_pad(
                frame_landmarks,
                original_width,
                original_height,
                target_width,
                target_height,
            )

            # Normalize landmarks to the target size
            frame_landmarks[0::3] /= target_width  # x coordinates
            frame_landmarks[1::3] /= target_height  # y coordinates

            frames.append(frame_landmarks)

    cap.release()
    return np.array(frames)


def normalize_landmarks_scale(
    flattened_frames, reference_scale=np.array([0.27518007, 0.68317342, 2.46986735])
):
    num_frames = flattened_frames.shape[0]
    num_landmarks = 161  # Each landmark has (x, y, z) coordinates

    # Reshape flattened frames to (num_frames, num_landmarks, 3)
    frames = flattened_frames.reshape(num_frames, num_landmarks, 3)

    normalized_frames = []

    for landmarks in frames:
        # Filter out landmarks with any zero value
        non_zero_landmarks = landmarks[~np.any(landmarks == 0, axis=1)]

        # Step 1: Find the center of the landmarks based on the specified method
        min_coords = np.min(non_zero_landmarks, axis=0)
        max_coords = np.max(non_zero_landmarks, axis=0)
        center = (min_coords + max_coords) / 2

        # Step 2: Translate landmarks to origin
        translated_landmarks = landmarks - center

        # Step 3: Normalize the landmarks to fit within the reference scale
        size = max_coords - min_coords
        normalized_landmarks = translated_landmarks / size * reference_scale

        # Step 4: Translate back to the center of the image (0.5, 0.5, 0.5)
        image_center = np.array([0.5, 0.5, 0.5])
        final_landmarks = normalized_landmarks + image_center

        # Replace zero-value landmarks with their original positions
        final_landmarks[landmarks == 0] = landmarks[landmarks == 0]

        normalized_frames.append(final_landmarks)

    # Reshape back to the original shape (num_frames, 483)
    normalized_flattened_frames = np.array(normalized_frames).reshape(
        num_frames, num_landmarks * 3
    )

    return normalized_flattened_frames


def forward_fill_landmarks(data):
    filled_data = pd.DataFrame(data)
    filled_data = filled_data.replace(to_replace=0, method="ffill")
    filled_data = np.array(filled_data)

    return filled_data


def pad_landmarks(landmarks, max_frames=512, pad_value=0):
    if landmarks.shape[0] < max_frames:
        padding = torch.full(
            (max_frames - landmarks.shape[0], landmarks.shape[1]), pad_value
        )
        padded_seq = torch.cat([landmarks, padding], dim=0)
        attention_mask = torch.cat(
            [
                torch.ones(landmarks.shape[0]),
                torch.zeros(max_frames - landmarks.shape[0]),
            ]
        )
    else:
        padded_seq = landmarks[:max_frames]
        attention_mask = torch.ones(max_frames)

    return padded_seq, attention_mask
