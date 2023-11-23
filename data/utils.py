import math
from collections import defaultdict
import numpy as np

import constants
import config

tssi_order = [
    3, 2, 19, 2, 1, 8, 10, 12, 10, 8, 1, 2,
    0, 7, 9, 11, 9, 7, 0, 2, 3, 6, 4, 14, 16,
    18, 16, 14, 4, 6, 5, 13, 15, 17, 15, 13,
    5, 6, 3, 2
]


def get_tssi_dim_sequence(dim_sequence):
    tssi_dim_sequence = []
    for frame_ind in range(dim_sequence.shape[0]):
        frame = []
        for joint in tssi_order:
            frame.append(dim_sequence[frame_ind][joint])
        tssi_dim_sequence.append(frame)
    return tssi_dim_sequence


def get_tssi_angle_sequence(sequence):
    
    sequence = np.reshape(sequence.copy(), (sequence.shape[0], -1))

    def get_angle(point1: np.ndarray, point2: np.ndarray) -> float:
        dot_product = np.dot(point1, point2)
        magnitude1 = np.linalg.norm(point1)
        magnitude2 = np.linalg.norm(point2)
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = math.degrees(angle_radians.item())
        angle_degrees /= 360.0
        if abs(angle_degrees) < 1e-8:
            return 0
        if math.isnan(angle_degrees):
            return 0
        return angle_degrees

    def get_augmented_frame(frame: np.ndarray) -> list:
        angles = []
        for i in tssi_order:
            joint1, joint2 = i, (i + 1) % constants.NUM_JOINTS
            point1 = frame[3 * joint1 : (3 * joint1 + 3)]
            point2 = frame[3 * joint2 : (3 * joint2 + 3)]
            angles.append(get_angle(point1=point1, point2=point2))
        return angles

    tssi_angle_sequence = []
    for frame_ind in range(sequence.shape[0]):
        tssi_angle_sequence.append(get_augmented_frame(frame=sequence[frame_ind]))
    return tssi_angle_sequence


def check_anomalies(data):
    for sequence in data:
        for frame in sequence:
            for joint in frame:
                if joint[0] >= 1000 or joint[1] >= 1000 or joint[2] >= 1000:
                    raise ValueError


def drop_indices(ls, indices):
    return [ls[i] for i in range(len(ls)) if i not in indices]


def remove_anomalies(data, labels, subjects):
    frames_to_drop = defaultdict(list)
    dropped = 0
    for sequence_ind, sequence in enumerate(data):
        for frame_ind in range(sequence.shape[0]):
            for joint in sequence[frame_ind]:
                assert joint.shape == (3,)
                if joint[0] >= 1000 or joint[1] >= 1000 or joint[2] >= 1000:
                    frames_to_drop[sequence_ind].append(frame_ind)
                    dropped += 1
    for k, v in frames_to_drop.items():
        dropped -= len(v)
    assert dropped == 0
    print("Dropping frames:", frames_to_drop)
    for sequence_ind in range(len(data)):
        data[sequence_ind] = np.delete(
            data[sequence_ind], frames_to_drop[sequence_ind], axis=0
        )
        
    check_anomalies(data)
    return data, labels, subjects

