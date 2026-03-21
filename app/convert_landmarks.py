from pathlib import Path
import json
import pandas as pd

LANDMARK_NAMES = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}


def main():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "output" / "landmarks.csv"
    named_csv_path = project_root / "output" / "landmarks_named.csv"
    json_path = project_root / "output" / "landmarks_by_frame.json"

    df = pd.read_csv(csv_path)
    df["landmark_name"] = df["landmark_index"].map(LANDMARK_NAMES)

    # 이름 붙인 CSV 저장
    df.to_csv(named_csv_path, index=False, encoding="utf-8-sig")

    # 프레임별 JSON 변환
    frames = []
    grouped = df.groupby("frame_index")

    for frame_index, group in grouped:
        joints = {}
        for _, row in group.iterrows():
            joint_name = row["landmark_name"]
            joints[joint_name] = {
                "x": float(row["world_x"]),
                "y": float(row["world_y"]),
                "z": float(row["world_z"]),
                "visibility": float(row["visibility"]),
            }

        frames.append({
            "frame_index": int(frame_index),
            "joints": joints
        })

    result = {"frames": frames}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"saved: {named_csv_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()