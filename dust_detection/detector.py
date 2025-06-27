import cv2
import numpy as np

def detect_lens_dust(image_path, min_area=30, blur_ksize=25, debug=False):
    """
    Phát hiện lỗi bụi/điểm đen trên ảnh camera.

    Args:
        image_path (str): Đường dẫn ảnh đầu vào.
        min_area (int): Diện tích nhỏ nhất để được coi là lỗi thật.
        blur_ksize (int): Kích thước kernel làm mờ Gaussian.
        debug (bool): Hiển thị các bước trung gian nếu True.

    Returns:
        result_img (ndarray): Ảnh đã vẽ khung vùng lỗi.
        defect_count (int): Số vùng lỗi được phát hiện.
        defect_coords (List[Tuple[int, int, int, int]]): Danh sách toạ độ (x, y, w, h) của các vùng lỗi.
    """

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Không thể đọc ảnh: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để mất chi tiết nhỏ
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

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
    defect_coords = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            defect_coords.append((x, y, w, h))

    if debug:
        cv2.imshow("🔍 Gray", gray)
        cv2.imshow("🔍 Blurred", blurred)
        cv2.imshow("🔍 Difference", diff)
        cv2.imshow("🔍 Thresholded", thresh)
        cv2.imshow("🔍 Closed", closed)

    return result_img, len(defect_coords), defect_coords