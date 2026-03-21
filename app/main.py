from pathlib import Path

from extractor import PoseExtractor


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "pose_landmarker.task"
    gif_path = project_root / "input" / "sample.gif"
    output_csv = project_root / "output" / "landmarks.csv"
    preview_dir = project_root / "output" / "frames"

    extractor = PoseExtractor(str(model_path))

    df = extractor.extract_from_gif(str(gif_path))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    extractor.export_preview_frames(str(gif_path), str(preview_dir), max_frames=12)

    print(f"saved csv: {output_csv}")
    print(f"saved preview frames: {preview_dir}")
    print(df.head())


if __name__ == "__main__":
    main()