import cv2
import numpy as np

def detect_lens_dust(image_path, min_area=30, debug=False):
    """
    Phát hiện lỗi bụi/điểm đen trên ảnh camera.

    Tham số:
        image_path (str): Đường dẫn ảnh đầu vào.
        min_area (int): Diện tích nhỏ nhất để được coi là lỗi thật.
        debug (bool): Nếu True, hiển thị ảnh trung gian để debug.

    Trả về:
        result_img (ndarray): Ảnh có vẽ khung vùng lỗi.
        num_defects (int): Số vùng lỗi được phát hiện.
    """

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để mất chi tiết nhỏ
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)

    # Trừ ảnh gốc với ảnh mờ → giữ lại chi tiết nhỏ (bụi)
    diff = cv2.absdiff(gray, blurred)

    # Threshold ảnh để tạo nhị phân
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Morphological closing để loại bỏ nhiễu nhỏ
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Tìm contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = image.copy()
    defect_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            defect_count += 1

    if debug:
        cv2.imshow("Gray", gray)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Difference", diff)
        cv2.imshow("Thresholded", thresh)
        cv2.imshow("Closed", closed)

    return result_img, defect_count
