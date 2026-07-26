def preprocess(img, input_size, swap=(2, 0, 1)):
    # Convert Torch tensor or other image types into a real NumPy array.
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    elif not isinstance(img, np.ndarray):
        img = np.asarray(img)

    # Convert CHW images to HWC.
    if (
        img.ndim == 3
        and img.shape[0] in (1, 3, 4)
        and img.shape[-1] not in (1, 3, 4)
    ):
        img = np.transpose(img, (1, 2, 0))

    # Remove alpha channel if present.
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Convert float images into uint8.
    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            img = img * 255.0

        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8, copy=False)

    # OpenCV requires contiguous NumPy memory.
    img = np.ascontiguousarray(img)

    if img.ndim == 3:
        padded_img = (
            np.ones(
                (input_size[0], input_size[1], 3),
                dtype=np.uint8,
            )
            * 114
        )
    else:
        padded_img = (
            np.ones(input_size, dtype=np.uint8) * 114
        )

    r = min(
        input_size[0] / img.shape[0],
        input_size[1] / img.shape[1],
    )

    resized_img = cv2.resize(
        img,
        (
            int(img.shape[1] * r),
            int(img.shape[0] * r),
        ),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    padded_img[
        : int(img.shape[0] * r),
        : int(img.shape[1] * r),
    ] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(
        padded_img,
        dtype=np.float32,
    )

    return padded_img, r
