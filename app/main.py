from pathlib import Path

from extractor import PoseExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "pose_landmarker.task"
    input_dir = project_root / "input"
    output_csv = project_root / "output" / "landmarks.csv"
    preview_dir = project_root / "output" / "previews"

    image_paths = sorted(
        str(path)
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"입력 폴더에 이미지가 없습니다: {input_dir} (지원 확장자: {sorted(IMAGE_EXTENSIONS)})"
        )

    extractor = PoseExtractor(str(model_path))

    df = extractor.extract_from_images(image_paths)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    extractor.export_preview_images(image_paths, str(preview_dir))

    print(f"processed images: {len(image_paths)}")
    print(f"saved csv: {output_csv}")
    print(f"saved previews: {preview_dir}")
    print(df.head())


if __name__ == "__main__":
    main()
