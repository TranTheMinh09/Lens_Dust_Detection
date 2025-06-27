import os
import cv2
from dust_detection.detector import detect_lens_dust
from colorama import Fore, Style, init

init(autoreset=True)

def main():
    image_folder = "images"
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not image_files:
        print(Fore.RED + "❌ Không tìm thấy ảnh trong thư mục images/")
        return

    print(Fore.CYAN + f"[INFO] Đang kiểm tra {len(image_files)} ảnh...\n")

    for filename in image_files:
        image_path = os.path.join(image_folder, filename)

        try:
            result_img, defect_count, defect_boxes = detect_lens_dust(image_path, debug=False)

            if defect_count == 0:
                status = Fore.GREEN + "✅ PASS"
            else:
                status = Fore.RED + f"❌ FAIL – Phát hiện {defect_count} lỗi"

            print(f"{filename}: {status}")

            # (Tuỳ chọn) Hiện toạ độ lỗi nếu có
            if defect_count > 0:
                for i, (x, y, w, h) in enumerate(defect_boxes, 1):
                    print(Fore.YELLOW + f"   └ Lỗi {i}: (x={x}, y={y}, w={w}, h={h})")

            # Resize ảnh
            resized_img = cv2.resize(result_img, (640, 480))
            cv2.imshow(f"Result - {filename}", resized_img)
            key = cv2.waitKey(2000) & 0xFF
            cv2.destroyAllWindows()

            if key == ord("q"):
                print(Fore.LIGHTRED_EX + "❎ Đã thoát sớm.")
                break

        except Exception as e:
            print(Fore.RED + f"[ERROR] {filename}: {e}")

if __name__ == "__main__":
    main()
