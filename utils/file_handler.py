import os
import re
import json
import zipfile
import shutil
import logging
import struct
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
from config import ALLOWED_EXTENSIONS, LOGS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "file_handler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FileHandler")

# Known dataset label mappings for automatic human-readable labeling
FASHION_MNIST_LABELS = {
    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle_Boot"
}

CIFAR10_LABELS = {
    0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4: "Deer",
    5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"
}

FER2013_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

def is_valid_image(image_path: Path) -> bool:
    """
    Verifies if an image is uncorrupted and readable.
    """
    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception as e:
        logger.warning(f"Corrupted or unreadable image found: {image_path}. Error: {e}")
        return False

def _resolve_label_name(lbl: Any, dataset_hint: str) -> str:
    """
    Maps raw label values (int, float, string) into clean, human-readable class names.
    """
    hint = dataset_hint.lower()
    try:
        lbl_int = int(lbl)
        if "fashion" in hint and lbl_int in FASHION_MNIST_LABELS:
            return FASHION_MNIST_LABELS[lbl_int]
        elif "cifar" in hint and lbl_int in CIFAR10_LABELS:
            return CIFAR10_LABELS[lbl_int]
        elif ("fer" in hint or "emotion" in hint) and lbl_int in FER2013_LABELS:
            return FER2013_LABELS[lbl_int]
        elif "mnist" in hint or "digit" in hint:
            return f"Digit_{lbl_int}"
        else:
            return f"Class_{lbl_int}"
    except (ValueError, TypeError):
        clean_name = str(lbl).strip().replace(" ", "_")
        # Remove unwanted punctuation
        clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '', clean_name)
        return clean_name or "Unknown_Class"

def detect_and_convert_csv_dataset(directory: Path, max_per_class: int = 60) -> Optional[Path]:
    """
    Detects if the directory contains a CSV-based image dataset:
    1. Multi-column pixel CSVs (Fashion-MNIST, MNIST, CIFAR, custom pixel datasets).
    2. Single-column space-separated pixel strings (e.g. FER-2013, facial expression CSVs).
    Automatically extracts a balanced subset into standard class image folders.
    """
    csv_files = list(directory.glob("*.csv")) + list(directory.glob("*/*.csv"))
    if not csv_files:
        return None

    # Prioritize train CSV if multiple exist
    csv_files.sort(key=lambda p: (0 if "train" in p.name.lower() else 1, -p.stat().st_size))

    for csv_file in csv_files:
        try:
            logger.info(f"Inspecting CSV candidate: {csv_file.name}")
            sample_df = pd.read_csv(csv_file, nrows=5)
            cols = list(sample_df.columns)
            hint = f"{csv_file.name}_{directory.name}"

            # --- CASE A: Single-column space-separated pixel string (FER2013 format) ---
            pixel_str_col = next((c for c in cols if str(c).lower() in ["pixels", "pixel", "image_data"]), None)
            if pixel_str_col and len(cols) <= 5:
                # Find label column
                lbl_candidates = [c for c in cols if c != pixel_str_col]
                if lbl_candidates:
                    label_col = lbl_candidates[0]
                    first_val = str(sample_df[pixel_str_col].iloc[0])
                    num_vals = len(first_val.split())
                    side = int(np.sqrt(num_vals))
                    if side * side == num_vals and side >= 4:
                        logger.info(f"Detected space-separated pixel string dataset ({side}x{side}) in {csv_file.name}")
                        out_dir = directory / "converted_images"
                        if out_dir.exists():
                            shutil.rmtree(out_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)

                        counts = {}
                        for chunk in pd.read_csv(csv_file, chunksize=500):
                            for _, row in chunk.iterrows():
                                cname = _resolve_label_name(row[label_col], hint)
                                if counts.get(cname, 0) >= max_per_class:
                                    continue
                                c_dir = out_dir / cname
                                c_dir.mkdir(parents=True, exist_ok=True)

                                raw_pixels = np.array(str(row[pixel_str_col]).split(), dtype=np.uint8).reshape((side, side))
                                img = Image.fromarray(raw_pixels)
                                idx = counts.get(cname, 0)
                                img.save(c_dir / f"{cname}_{idx:04d}.png")
                                counts[cname] = idx + 1

                            if len(counts) >= 2 and all(v >= max_per_class for v in counts.values()):
                                break
                            if sum(counts.values()) >= 1500:
                                break

                        if len(counts) >= 2:
                            logger.info(f"Successfully converted FER-style pixel CSV into {len(counts)} classes at {out_dir}")
                            return out_dir

            # --- CASE B: Multi-column pixel dataset ---
            # Find label column
            label_col = None
            for c in cols:
                if str(c).lower() in ["label", "target", "class", "category", "y", "emotion", "diagnosis"]:
                    label_col = c
                    break
            if label_col is None and len(cols) > 65:
                # Check first column or last column
                for candidate_col in [cols[0], cols[-1]]:
                    if sample_df[candidate_col].nunique() <= 100:
                        label_col = candidate_col
                        break

            if not label_col:
                continue

            pixel_cols = [c for c in cols if c != label_col]
            num_pixels = len(pixel_cols)
            if num_pixels < 64:
                continue

            # Check dimensions (Grayscale square or RGB)
            is_rgb = False
            img_h, img_w = 0, 0
            if num_pixels % 3 == 0:
                side_rgb = int(np.sqrt(num_pixels // 3))
                if side_rgb * side_rgb == (num_pixels // 3):
                    img_h, img_w = side_rgb, side_rgb
                    is_rgb = True

            if not is_rgb:
                side = int(np.sqrt(num_pixels))
                if side * side == num_pixels:
                    img_h, img_w = side, side
                else:
                    # Check standard rectangular dimensions
                    for candidate_h in [28, 32, 48, 64, 96, 128]:
                        if num_pixels % candidate_h == 0:
                            candidate_w = num_pixels // candidate_h
                            if 0.5 <= candidate_w / candidate_h <= 2.0:
                                img_h, img_w = candidate_h, candidate_w
                                break
                    if img_h == 0:
                        continue

            logger.info(f"Detected pixel image CSV ({img_w}x{img_h}, rgb={is_rgb}) in {csv_file.name}")
            out_dir = directory / "converted_images"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            counts = {}
            for chunk in pd.read_csv(csv_file, chunksize=500):
                for _, row in chunk.iterrows():
                    cname = _resolve_label_name(row[label_col], hint)
                    if counts.get(cname, 0) >= max_per_class:
                        continue

                    c_dir = out_dir / cname
                    c_dir.mkdir(parents=True, exist_ok=True)

                    raw_vals = row[pixel_cols].values.astype(np.uint8)
                    if is_rgb:
                        img_arr = raw_vals.reshape((img_h, img_w, 3))
                    else:
                        img_arr = raw_vals.reshape((img_h, img_w))

                    img = Image.fromarray(img_arr)
                    idx = counts.get(cname, 0)
                    img.save(c_dir / f"{cname}_{idx:04d}.png")
                    counts[cname] = idx + 1

                if len(counts) >= 2 and all(v >= max_per_class for v in counts.values()):
                    break
                if sum(counts.values()) >= 1500:
                    break

            if len(counts) >= 2:
                logger.info(f"Successfully converted CSV dataset into {len(counts)} classes at {out_dir}")
                return out_dir
        except Exception as e:
            logger.warning(f"Error parsing CSV {csv_file}: {e}")

    return None

def detect_and_organize_csv_indexed_images(directory: Path, max_per_class: int = 100) -> Optional[Path]:
    """
    Detects Kaggle-style datasets where a CSV contains image filename/ID and class label
    (e.g., train.csv + images folder), and organizes them into class subdirectories.
    """
    csv_files = list(directory.glob("*.csv")) + list(directory.glob("*/*.csv"))
    if not csv_files:
        return None

    # Collect all image files in directory
    all_images = [f for f in directory.rglob('*') if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
    if not all_images:
        return None

    image_name_map = {f.name: f for f in all_images}
    image_stem_map = {f.stem: f for f in all_images}

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, nrows=100)
            cols = list(df.columns)
            
            # Find image filename / ID column
            img_col = next((c for c in cols if any(k in str(c).lower() for k in ["image", "file", "id", "name", "path"])), None)
            # Find label / class column
            lbl_col = next((c for c in cols if c != img_col and any(k in str(c).lower() for k in ["label", "target", "class", "category", "diagnosis", "disease", "type"])), None)

            if not img_col or not lbl_col:
                continue

            # Verify if values match existing files
            sample_ids = df[img_col].astype(str).tolist()[:20]
            matches = sum(1 for sid in sample_ids if sid in image_name_map or sid in image_stem_map)
            if matches < 3:
                continue

            logger.info(f"Detected CSV-indexed dataset in {csv_file.name} (img_col='{img_col}', lbl_col='{lbl_col}')")
            out_dir = directory / "organized_by_class"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Read full CSV and organize
            full_df = pd.read_csv(csv_file)
            hint = f"{csv_file.name}_{directory.name}"
            counts = {}

            for _, row in full_df.iterrows():
                sid = str(row[img_col]).strip()
                target_file = image_name_map.get(sid) or image_stem_map.get(sid)
                if not target_file or not target_file.exists():
                    continue

                cname = _resolve_label_name(row[lbl_col], hint)
                if counts.get(cname, 0) >= max_per_class:
                    continue

                c_dir = out_dir / cname
                c_dir.mkdir(parents=True, exist_ok=True)

                dest_file = c_dir / target_file.name
                if not dest_file.exists():
                    shutil.copy(str(target_file), str(dest_file))
                counts[cname] = counts.get(cname, 0) + 1

            if len(counts) >= 2:
                logger.info(f"Successfully organized CSV-indexed dataset into {len(counts)} classes at {out_dir}")
                return out_dir
        except Exception as e:
            logger.warning(f"Error checking CSV index {csv_file}: {e}")

    return None

def detect_and_organize_json_annotations(directory: Path, max_per_class: int = 100) -> Optional[Path]:
    """
    Detects JSON or text annotation files mapping images to labels (e.g. annotations.json).
    """
    json_files = list(directory.glob("*.json")) + list(directory.glob("*/*.json"))
    all_images = [f for f in directory.rglob('*') if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
    if not all_images or not json_files:
        return None

    image_name_map = {f.name: f for f in all_images}
    image_stem_map = {f.stem: f for f in all_images}

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            mapping = {}
            if isinstance(data, dict):
                # Format: {"img1.jpg": "cat", ...}
                mapping = data
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # Format: [{"image": "img1.jpg", "label": "cat"}, ...]
                first_item = data[0]
                img_key = next((k for k in first_item.keys() if any(w in k.lower() for w in ["image", "file", "id", "name"])), None)
                lbl_key = next((k for k in first_item.keys() if k != img_key and any(w in k.lower() for w in ["label", "class", "target", "category"])), None)
                if img_key and lbl_key:
                    for item in data:
                        mapping[item[img_key]] = item[lbl_key]

            if not mapping:
                continue

            # Check matches
            matches = sum(1 for k in list(mapping.keys())[:15] if str(k) in image_name_map or str(k) in image_stem_map)
            if matches < 3:
                continue

            logger.info(f"Detected JSON-indexed dataset in {jf.name}")
            out_dir = directory / "organized_by_class"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            counts = {}
            for img_name, label in mapping.items():
                target_file = image_name_map.get(str(img_name)) or image_stem_map.get(str(img_name))
                if not target_file or not target_file.exists():
                    continue

                cname = _resolve_label_name(label, jf.name)
                if counts.get(cname, 0) >= max_per_class:
                    continue

                c_dir = out_dir / cname
                c_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(target_file), str(c_dir / target_file.name))
                counts[cname] = counts.get(cname, 0) + 1

            if len(counts) >= 2:
                logger.info(f"Successfully organized JSON-indexed dataset into {len(counts)} classes at {out_dir}")
                return out_dir
        except Exception as e:
            logger.warning(f"Error checking JSON annotations {jf}: {e}")

    return None

def detect_and_convert_ubyte_dataset(directory: Path, max_per_class: int = 60) -> Optional[Path]:
    """
    Detects if the directory contains raw IDX ubyte files (e.g. MNIST/Fashion-MNIST ubytes)
    and converts a balanced subset into standard class folders.
    """
    img_files = list(directory.glob("*images*idx3-ubyte*")) + list(directory.glob("*images*ubyte*"))
    lbl_files = list(directory.glob("*labels*idx1-ubyte*")) + list(directory.glob("*labels*ubyte*"))
    
    if not img_files or not lbl_files:
        return None

    img_file = img_files[0]
    lbl_file = lbl_files[0]

    try:
        with open(img_file, 'rb') as f_img, open(lbl_file, 'rb') as f_lbl:
            magic_img, num_imgs, rows, cols = struct.unpack(">IIII", f_img.read(16))
            magic_lbl, num_lbls = struct.unpack(">II", f_lbl.read(8))

            if magic_img != 2051 or magic_lbl != 2049:
                return None

            out_dir = directory / "converted_ubyte_images"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            hint = f"{img_file.name}_{directory.name}"
            counts = {}
            img_size = rows * cols

            for _ in range(min(num_imgs, 2000)):
                lbl_byte = f_lbl.read(1)
                img_bytes = f_img.read(img_size)
                if not lbl_byte or len(img_bytes) < img_size:
                    break

                class_name = _resolve_label_name(lbl_byte[0], hint)
                if counts.get(class_name, 0) >= max_per_class:
                    continue

                class_folder = out_dir / class_name
                class_folder.mkdir(parents=True, exist_ok=True)

                img_arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((rows, cols))
                img = Image.fromarray(img_arr)
                idx = counts.get(class_name, 0)
                img.save(class_folder / f"{class_name}_{idx:04d}.png")
                counts[class_name] = idx + 1

                if len(counts) >= 2 and all(v >= max_per_class for v in counts.values()):
                    break

            if len(counts) >= 2:
                logger.info(f"Successfully converted ubyte dataset into {len(counts)} classes at {out_dir}")
                return out_dir
    except Exception as e:
        logger.warning(f"Error parsing ubyte files: {e}")

    return None

def auto_organize_flat_images(directory: Path) -> Optional[Path]:
    """
    If images are placed flat inside a directory without subfolders,
    checks if filenames contain class patterns:
    - 'cat_001.jpg' -> 'cat'
    - 'Normal-12.png' -> 'Normal'
    - 'benign (1).jpg' -> 'benign'
    - 'cat.1.jpg' -> 'cat'
    Organizes them into clean class subdirectories automatically.
    """
    images = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
    if len(images) < 6:
        return None

    # Analyze filename prefix distributions
    prefix_counts = {}
    file_prefix_map = {}

    for img in images:
        name = img.stem
        # 1. Clean trailing numbers or parenthesized indices like (1) or _001 or -02 or .3
        clean = re.sub(r'[\s_.-]*\(\d+\)$', '', name)
        clean = re.sub(r'[\s_.-]+\d+$', '', clean)

        # 2. If cleaned is different and non-empty, use as class candidate
        if clean and not clean.isdigit() and len(clean) >= 2:
            candidate = clean
        else:
            # Fallback: take first part before separator
            candidate = None
            for sep in ['.', '_', '-', ' ']:
                parts = name.split(sep)
                if len(parts) >= 2 and not parts[0].isdigit() and len(parts[0]) >= 2:
                    candidate = parts[0]
                    break

        if candidate:
            candidate_clean = candidate.strip().title()
            prefix_counts[candidate_clean] = prefix_counts.get(candidate_clean, 0) + 1
            file_prefix_map[img] = candidate_clean

    valid_prefixes = [p for p, count in prefix_counts.items() if count >= 3]
    if len(valid_prefixes) >= 2:
        logger.info(f"Auto-organizing flat images by detected class patterns: {valid_prefixes}")
        for img, p in file_prefix_map.items():
            if p in valid_prefixes:
                target_dir = directory / p
                target_dir.mkdir(parents=True, exist_ok=True)
                dest = target_dir / img.name
                if not dest.exists():
                    shutil.move(str(img), str(dest))
        return directory

    return None

def find_dataset_root(directory: Path) -> Path:
    """
    Recursively inspects directory to discover the true dataset root for ANY image dataset:
    1. Direct class subdirectories
    2. CSV pixel datasets (e.g. Fashion-MNIST, MNIST, CIFAR, custom CSVs)
    3. CSV indexed image datasets (e.g. train.csv + image folder)
    4. JSON annotation indexed image datasets
    5. IDX ubyte binary datasets
    6. Nested class subdirectories across arbitrary depths (e.g., train/cats, dataset/v1/classes)
    7. Flat images with class naming patterns in filenames
    """
    if not directory.exists():
        return directory

    # 1. Check if directly valid (folder contains >=2 subdirs with image files)
    direct_subdirs = [d for d in directory.iterdir() if d.is_dir() and not d.name.startswith(('.', '_'))]
    valid_direct_classes = 0
    for d in direct_subdirs:
        if any(f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS for f in d.iterdir()):
            valid_direct_classes += 1
    if valid_direct_classes >= 2:
        return directory

    # 2. Check for CSV-based pixel dataset
    csv_converted = detect_and_convert_csv_dataset(directory)
    if csv_converted:
        return csv_converted

    # 3. Check for CSV-indexed image dataset (train.csv + image folder)
    csv_indexed = detect_and_organize_csv_indexed_images(directory)
    if csv_indexed:
        return csv_indexed

    # 4. Check for JSON-indexed image dataset
    json_indexed = detect_and_organize_json_annotations(directory)
    if json_indexed:
        return json_indexed

    # 5. Check for IDX ubyte dataset
    ubyte_converted = detect_and_convert_ubyte_dataset(directory)
    if ubyte_converted:
        return ubyte_converted

    # 6. Search recursively for nested class folders
    candidates = []
    for d in directory.rglob('*'):
        if d.is_dir() and not d.name.startswith(('.', '_')):
            child_subdirs = [c for c in d.iterdir() if c.is_dir() and not c.name.startswith(('.', '_'))]
            if len(child_subdirs) >= 2:
                total_imgs = 0
                valid_children = 0
                for c in child_subdirs:
                    imgs = [f for f in c.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
                    if imgs:
                        valid_children += 1
                        total_imgs += len(imgs)
                if valid_children >= 2:
                    is_train = "train" in d.name.lower()
                    candidates.append((d, total_imgs, is_train))

    if candidates:
        # Prefer folders named 'train', then by maximum image count
        candidates.sort(key=lambda x: (1 if x[2] else 0, x[1]), reverse=True)
        logger.info(f"Resolved nested dataset root to: {candidates[0][0]} with {candidates[0][1]} images")
        return candidates[0][0]

    # 7. Check if flat images can be organized by filename conventions
    organized = auto_organize_flat_images(directory)
    if organized:
        return organized

    return directory

def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """
    Extracts a zip file and automatically resolves the true dataset root.
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Clean up OS specific metadata folders
    for metadata_dir in ['__MACOSX', '.DS_Store', 'Thumbs.db']:
        for meta_path in extract_to.rglob(metadata_dir):
            if meta_path.exists():
                if meta_path.is_dir():
                    shutil.rmtree(meta_path)
                else:
                    meta_path.unlink()

    # Intelligently locate, convert, or index dataset root
    resolved_root = find_dataset_root(extract_to)
    logger.info(f"Final resolved dataset root after extraction: {resolved_root}")
    return resolved_root

def validate_dataset_structure(dataset_path: Path) -> Dict[str, Any]:
    """
    Validates dataset structure and automatically resolves nested, indexed, or CSV datasets.
    """
    logger.info(f"Validating dataset at {dataset_path}")
    report = {
        "valid": False,
        "classes": [],
        "class_counts": {},
        "corrupted_files": [],
        "invalid_format_files": [],
        "total_images": 0,
        "errors": [],
        "resolved_path": str(dataset_path)
    }

    if not dataset_path.exists():
        report["errors"].append("Dataset path does not exist.")
        return report

    # Auto-resolve if current path lacks subdirectories
    resolved_path = find_dataset_root(dataset_path)
    if resolved_path != dataset_path and resolved_path.exists():
        dataset_path = resolved_path
        report["resolved_path"] = str(resolved_path)

    subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(('.', '_'))]
    
    if not subdirs:
        root_images = [f for f in dataset_path.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
        if root_images:
            report["errors"].append(
                "Images found directly in root folder without class subdirectories. "
                "Please organize images into subfolders by class (e.g., 'cats/', 'dogs/')."
            )
        else:
            report["errors"].append("No class subdirectories or image files found in the dataset archive.")
        return report

    valid_classes = []
    class_counts = {}
    corrupted_images = []
    invalid_format_files = []
    total_images = 0

    for class_dir in subdirs:
        class_name = class_dir.name
        images_in_class = []
        for file in class_dir.iterdir():
            if file.is_file():
                ext = file.suffix.lower()
                if ext in ALLOWED_EXTENSIONS:
                    if is_valid_image(file):
                        images_in_class.append(file)
                    else:
                        corrupted_images.append(str(file.relative_to(dataset_path)))
                elif not file.name.startswith('.'):
                    invalid_format_files.append(str(file.relative_to(dataset_path)))
        
        if images_in_class:
            valid_classes.append(class_name)
            class_counts[class_name] = len(images_in_class)
            total_images += len(images_in_class)
        else:
            logger.warning(f"Empty class folder found: {class_name}")

    report["classes"] = valid_classes
    report["class_counts"] = class_counts
    report["corrupted_files"] = corrupted_images
    report["invalid_format_files"] = invalid_format_files
    report["total_images"] = total_images

    # Validation criteria: at least 2 classes and some minimum number of total images
    if len(valid_classes) < 2:
        report["errors"].append(f"Classification benchmarking requires at least 2 class folders. Found {len(valid_classes)}: {valid_classes}")
    elif total_images < 6:
        report["errors"].append(f"Too few images to train and split. Found a total of {total_images} valid images across all classes. Minimum is 6.")
    else:
        report["valid"] = True

    logger.info(f"Validation report: Valid={report['valid']}, Classes={valid_classes}, Total Images={total_images}")
    return report

def generate_shapes_dataset(dest_dir: Path) -> Path:
    """
    Generates a small shapes dataset (Circles vs Squares) for testing and out-of-the-box demo validation.
    Includes intentional noise, duplicate hashes, and blurs to test the analysis engine.
    """
    logger.info(f"Generating synthetic shapes dataset at {dest_dir}")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
        
    classes = ["circle", "square"]
    num_samples = 30 # 30 circles, 30 squares
    
    for c in classes:
        (dest_dir / c).mkdir(parents=True, exist_ok=True)
        
    for i in range(num_samples):
        # 1. Circle class creation
        canvas_c = (np.random.normal(120, 5, (128, 128, 3))).astype(np.uint8)
        center_x = 64 + np.random.randint(-12, 12)
        center_y = 64 + np.random.randint(-12, 12)
        radius = np.random.randint(22, 38)
        color_c = (np.random.randint(200, 256), np.random.randint(30, 80), np.random.randint(80, 150))
        cv2.circle(canvas_c, (center_x, center_y), radius, color_c, -1)
        
        # 2. Square class creation
        canvas_s = (np.random.normal(120, 5, (128, 128, 3))).astype(np.uint8)
        w = np.random.randint(20, 35)
        x0 = 64 - w + np.random.randint(-8, 8)
        y0 = 64 - w + np.random.randint(-8, 8)
        x1 = 64 + w + np.random.randint(-8, 8)
        y1 = 64 + w + np.random.randint(-8, 8)
        color_s = (np.random.randint(30, 80), np.random.randint(120, 200), np.random.randint(200, 256))
        cv2.rectangle(canvas_s, (x0, y0), (x1, y1), color_s, -1)

        # Apply perturbations
        if i in [5, 10]:
            canvas_c = cv2.GaussianBlur(canvas_c, (15, 15), 0)
            canvas_s = cv2.GaussianBlur(canvas_s, (15, 15), 0)
            
        if i in [8, 18]:
            noise_c = np.random.normal(0, 25, canvas_c.shape).astype(np.int16)
            noise_s = np.random.normal(0, 25, canvas_s.shape).astype(np.int16)
            canvas_c = np.clip(canvas_c.astype(np.int16) + noise_c, 0, 255).astype(np.uint8)
            canvas_s = np.clip(canvas_s.astype(np.int16) + noise_s, 0, 255).astype(np.uint8)

        # Save images
        cv2.imwrite(str(dest_dir / "circle" / f"circle_{i}.jpg"), canvas_c)
        if i == 25:
            shutil.copy(str(dest_dir / "circle" / "circle_24.jpg"), str(dest_dir / "circle" / "circle_25.jpg"))
            cv2.imwrite(str(dest_dir / "square" / f"square_{i}.jpg"), canvas_s)
        else:
            cv2.imwrite(str(dest_dir / "square" / f"square_{i}.jpg"), canvas_s)

    logger.info("Shapes dataset generation complete.")
    return dest_dir
