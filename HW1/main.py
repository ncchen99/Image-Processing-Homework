import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QGridLayout, QGroupBox, QLineEdit, QHBoxLayout, QSlider, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from scipy import ndimage


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw1 - 念誠")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        # Load Image Buttons
        self.load_image_layout = QVBoxLayout()
        self.load_image1_button = QPushButton("Load Image 1")
        self.load_image1_button.clicked.connect(self.load_image1)
        self.load_image_layout.addWidget(self.load_image1_button)

        self.load_image2_button = QPushButton("Load Image 2")
        self.load_image2_button.clicked.connect(self.load_image2)
        self.load_image_layout.addWidget(self.load_image2_button)

        self.layout.addLayout(self.load_image_layout, 0, 0)

        # Group 1: Image Processing
        self.group1 = QGroupBox("1. Image Processing")
        self.group1_layout = QVBoxLayout()
        self.group1.setLayout(self.group1_layout)

        self.color_separation_button = QPushButton("1.1 Color Separation")
        self.color_separation_button.clicked.connect(self.color_separation)
        self.group1_layout.addWidget(self.color_separation_button)

        self.color_transformation_button = QPushButton(
            "1.2 Color Transformation")
        self.color_transformation_button.clicked.connect(
            self.color_transformation)
        self.group1_layout.addWidget(self.color_transformation_button)

        self.color_extraction_button = QPushButton("1.3 Color Extraction")
        self.color_extraction_button.clicked.connect(self.color_extraction)
        self.group1_layout.addWidget(self.color_extraction_button)

        self.layout.addWidget(self.group1, 0, 1)

        # Group 2: Image Smoothing
        self.group2 = QGroupBox("2. Image Smoothing")
        self.group2_layout = QVBoxLayout()
        self.group2.setLayout(self.group2_layout)

        self.gaussian_blur_button = QPushButton("2.1 Gaussian Blur")
        self.gaussian_blur_button.clicked.connect(self.gaussian_blur)
        self.group2_layout.addWidget(self.gaussian_blur_button)

        self.bilateral_filter_button = QPushButton("2.2 Bilateral Filter")
        self.bilateral_filter_button.clicked.connect(self.bilateral_filter)
        self.group2_layout.addWidget(self.bilateral_filter_button)

        self.median_filter_button = QPushButton("2.3 Median Filter")
        self.median_filter_button.clicked.connect(self.median_filter)
        self.group2_layout.addWidget(self.median_filter_button)

        self.layout.addWidget(self.group2, 1, 1)

        # Group 3: Edge Detection
        self.group3 = QGroupBox("3. Edge Detection")
        self.group3_layout = QVBoxLayout()
        self.group3.setLayout(self.group3_layout)

        self.sobel_x_button = QPushButton("3.1 Sobel X")
        self.sobel_x_button.clicked.connect(self.sobel_x)
        self.group3_layout.addWidget(self.sobel_x_button)

        self.sobel_y_button = QPushButton("3.2 Sobel Y")
        self.sobel_y_button.clicked.connect(self.sobel_y)
        self.group3_layout.addWidget(self.sobel_y_button)

        self.combination_threshold_button = QPushButton(
            "3.3 Combination and Threshold")
        self.combination_threshold_button.clicked.connect(
            self.combination_and_threshold)
        self.group3_layout.addWidget(self.combination_threshold_button)

        self.gradient_angle_button = QPushButton("3.4 Gradient Angle")
        self.gradient_angle_button.clicked.connect(self.gradient_angle)
        self.group3_layout.addWidget(self.gradient_angle_button)

        self.layout.addWidget(self.group3, 2, 1)
        
        
        # 初始化 Transformation controls
        self.angle_input = QLineEdit()
        self.scale_input = QLineEdit()
        self.tx_input = QLineEdit()
        self.ty_input = QLineEdit()

        # Group 4: Transforms
        self.group4 = QGroupBox("4. Transforms")
        self.group4_layout = QVBoxLayout()
        self.group4.setLayout(self.group4_layout)

        self.group4_layout.addWidget(QLabel("Rotation (degrees):"))
        self.group4_layout.addWidget(self.angle_input)

        self.group4_layout.addWidget(QLabel("Scaling (factor):"))
        self.group4_layout.addWidget(self.scale_input)

        self.group4_layout.addWidget(QLabel("Translation Tx:"))
        self.group4_layout.addWidget(self.tx_input)

        self.group4_layout.addWidget(QLabel("Translation Ty:"))
        self.group4_layout.addWidget(self.ty_input)

        self.transform_button = QPushButton("4. Transforms")
        self.transform_button.clicked.connect(self.apply_transforms)
        self.group4_layout.addWidget(self.transform_button)

        self.layout.addWidget(self.group4, 0, 2)

        # Group 6: Image Preview
        self.group6 = QGroupBox("6. Image Preview")
        self.group6_layout = QHBoxLayout()
        self.group6.setLayout(self.group6_layout)

        self.image1_label = QLabel("No Image 1 Loaded")
        self.image1_label.setMaximumHeight(250)
        self.group6_layout.addWidget(self.image1_label)

        self.image2_label = QLabel("No Image 2 Loaded")
        self.image2_label.setMaximumHeight(250)
        self.group6_layout.addWidget(self.image2_label)

        self.layout.addWidget(self.group6, 1, 2)

        # Image data
        self.image1 = None
        self.image2 = None
        
        self.sobel_processed_using_abs = True

    def load_image1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image1 = cv2.imread(file_path)
            self.display_image(self.image1, self.image1_label)

    def load_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image2 = cv2.imread(file_path)
            self.display_image(self.image2, self.image2_label)

    def display_image(self, img, label):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(
            img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(out_image))
        label.setAlignment(Qt.AlignCenter)

    # 1.1 Color Separation
    def color_separation(self):
        if self.image1 is not None:
            b, g, r = cv2.split(self.image1)
            zeros = np.zeros_like(b)
            b_image = cv2.merge([b, zeros, zeros])
            g_image = cv2.merge([zeros, g, zeros])
            r_image = cv2.merge([zeros, zeros, r])
            cv2.imshow("Blue Channel", b_image)
            cv2.imshow("Green Channel", g_image)
            cv2.imshow("Red Channel", r_image)

    # 1.2 Color Transformation
    def color_transformation(self):
        if self.image1 is not None:
            gray_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Gray Image (OpenCV)", gray_image)
            b, g, r = cv2.split(self.image1)
            avg_gray = ((b + g + r) / 3).astype(np.uint8)
            cv2.imshow("Gray Image (Average)", avg_gray)

    # 1.3 Color Extraction
    def color_extraction(self):
        if self.image1 is not None:
            hsv_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([18, 0, 25])
            upper_bound = np.array([85, 255, 255])
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            mask_inverse = cv2.bitwise_not(mask)
            extracted_image = cv2.bitwise_and(
                self.image1, self.image1, mask=mask_inverse)
            cv2.imshow("Yellow-Green Mask", mask)
            cv2.imshow("Image without Yellow-Green", extracted_image)

    def show_filter_window(self, filter_function):
        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Preview")
        layout = QVBoxLayout(dialog)

        image_label = QLabel()
        layout.addWidget(image_label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(5)
        slider.setValue(0)
        layout.addWidget(slider)

        def update_image(value):
            filter_function(value, image_label)

        slider.valueChanged.connect(update_image)
        update_image(0)  # Initialize with default value

        dialog.exec_()

    # 2.1 Gaussian Blur
    def gaussian_blur(self):
        if self.image1 is not None:
            self.show_filter_window(self.apply_gaussian_blur)

    def apply_gaussian_blur(self, m, label):
        kernel_size = (2 * m + 1, 2 * m + 1)
        blur_image = cv2.GaussianBlur(self.image1, kernel_size, 0)
        self.display_image(blur_image, label)

    # 2.2 Bilateral Filter
    def bilateral_filter(self):
        if self.image1 is not None:
            self.show_filter_window(self.apply_bilateral_filter)

    def apply_bilateral_filter(self, m, label):
        d = 2 * m + 1
        bilateral_image = cv2.bilateralFilter(self.image1, d, 90, 90)
        self.display_image(bilateral_image, label)

    # 2.3 Median Filter
    def median_filter(self):
        if self.image1 is not None:
            self.show_filter_window(self.apply_median_filter)

    def apply_median_filter(self, m, label):
        kernel_size = 2 * m + 1
        median_image = cv2.medianBlur(self.image1, kernel_size)
        self.display_image(median_image, label)


    def apply_sobel_operator(self, padded_image, operator):
        # Get the dimensions of the grayscale image
        height, width = padded_image.shape[0] - 2, padded_image.shape[1] - 2
        
        # Create an empty output image
        output = np.zeros((height, width), dtype=np.float32)
        
        # Apply the Sobel operator using convolution
        for i in range(height):
            for j in range(width):
                # Extract the 3x3 region
                region = padded_image[i:i+3, j:j+3]
                # Apply the operator (convolution)
                result = np.sum(region * operator)
                if self.sobel_processed_using_abs:
                    output[i, j] = abs(result)  # Take the absolute value
                else:
                    output[i, j] = result

        # Convert the output to 8-bit format
        output = cv2.convertScaleAbs(output)
        return output

    def process_sobel_x(self, gray):
        # Apply Sobel X
        # Apply Gaussian smoothing with kernel size 5x5 and sigmaX = 0
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Define Sobel X operator
        sobel_x_operator = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=np.float32)

        # Apply Sobel X operator using custom convolution
        sobelx = self.apply_sobel_operator(blur, sobel_x_operator)
        return sobelx

    def process_sobel_y(self, gray):
        # Apply Sobel Y
        # Apply Gaussian smoothing with kernel size 5x5 and sigmaX = 0
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Define Sobel Y operator
        sobel_y_operator = np.array([[-1, -2, -1],
                                    [0,  0,  0],
                                    [1,  2,  1]], dtype=np.float32)
        # Apply Sobel Y operator using custom convolution
        sobely = self.apply_sobel_operator(blur, sobel_y_operator)
        return sobely
    
    # 3.1 Sobel X
    def sobel_x(self):
        if self.image1 is not None:
            gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            sobelx = self.process_sobel_x(gray)
            # Show the result
            cv2.imshow("Sobel X", sobelx)

    # 3.2 Sobel Y
    def sobel_y(self):
        if self.image1 is not None:
            gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            sobely = self.process_sobel_y(gray)
            # Show the result
            cv2.imshow("Sobel Y", sobely)

    

    # 3.3 Combination and Threshold
    def combination_and_threshold(self):
        if self.image1 is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)

            # Apply Sobel X and Y
            sobelx = self.process_sobel_x(gray)
            sobely = self.process_sobel_y(gray)

            # Combine Sobel X and Y
            combination = np.hypot(sobelx, sobely).astype(np.uint8)
            combination = cv2.convertScaleAbs(combination)

            # Normalize the result to 0-255
            normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply threshold with value 128
            _, result1 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)

            # Apply threshold with value 28
            _, result2 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)

            # Show the results
            cv2.imshow("Combination", normalized)
            cv2.imshow("Threshold 128", result1.astype(np.uint8))
            cv2.imshow("Threshold 28", result2.astype(np.uint8))
            
    # 3.4 Gradient Angle
    def gradient_angle(self):
        if self.image1 is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            self.sobel_processed_using_abs = False
            # Process Sobel X and Y
            sobelx = self.process_sobel_x(gray)
            sobely = self.process_sobel_y(gray)
            self.sobel_processed_using_abs = True
            combination = np.hypot(sobelx, sobely).astype(np.uint8)
            combination = cv2.convertScaleAbs(combination)
            normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX)

            # Calculate gradient angle
            angle = np.degrees(np.arctan2(sobely, sobelx))
            angle = (angle + 360) % 360

            # Generate masks for angle ranges
            mask1 = (((angle >= 170) & (angle <= 190))).astype(np.uint8) * 255
            mask2 = ((angle >= 260) & (angle <= 280)).astype(np.uint8) * 255
            
            # Apply bitwise AND to get the results
            result1 = cv2.bitwise_and(normalized, mask1)
            result2 = cv2.bitwise_and(normalized, mask2)

            # Show the results
            cv2.imshow("Gradient Angle 170-190", result1)
            cv2.imshow("Gradient Angle 260-280", result2)
    def apply_transforms(self):
        if self.image1 is not None:
            # 讀取旋轉角度
            try:
                angle = float(self.angle_input.text())
            except ValueError:
                angle = 0  # 預設為 0 度

            # 讀取縮放比例
            try:
                scale = float(self.scale_input.text())
            except ValueError:
                scale = 1.0  # 預設比例為 1.0

            # 讀取平移量 Tx 和 Ty
            try:
                tx = float(self.tx_input.text())
                ty = float(self.ty_input.text())
            except ValueError:
                tx, ty = 0, 0  # 預設平移量為 0

            # 影像中心
            h, w = self.image1.shape[:2]
            center = (w // 2, h // 2)

            # 4.1 旋轉矩陣
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

            # 先進行旋轉和縮放
            rotated_image = cv2.warpAffine(self.image1, rotation_matrix, (w, h))

            # 4.3 平移矩陣
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty - 400]])

            # 再進行平移
            transformed_image = cv2.warpAffine(rotated_image, translation_matrix, (w, h))

            # 顯示結果
            cv2.namedWindow("Transformed Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Transformed Image", transformed_image)
            cv2.resizeWindow("Transformed Image", 800, 600)  # 調整到你想要的大小

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
