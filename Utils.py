import numpy as np


def parse_int32(byte_list: bytes, start_i: int) -> int:
    """Parse a big-endian int32 value from a byte sequence."""
    return (
        (byte_list[start_i] << 24)
        | (byte_list[start_i + 1] << 16)
        | (byte_list[start_i + 2] << 8)
        | byte_list[start_i + 3]
    )


def parse_dataset(
    img_filepath: str,
    label_filepath: str,
) -> tuple[int, int, int, np.ndarray, int, np.ndarray]:
    """Load the MNIST-style dataset into NumPy arrays."""
    with open(img_filepath, "rb") as file_obj:
        img_raw = file_obj.read()

    num_img = parse_int32(img_raw, 4)
    n_r = parse_int32(img_raw, 8)
    n_c = parse_int32(img_raw, 12)
    img_data = np.frombuffer(img_raw, dtype=np.uint8, offset=16)
    imgs = img_data.reshape(num_img, n_r, n_c).copy()

    with open(label_filepath, "rb") as file_obj:
        label_raw = file_obj.read()

    num_label = parse_int32(label_raw, 4)
    labels = np.frombuffer(label_raw, dtype=np.uint8, offset=8).copy()

    if num_img != num_label:
        raise ValueError(
            f"Image count {num_img} does not match label count {num_label}."
        )

    return num_img, n_r, n_c, imgs, num_label, labels
