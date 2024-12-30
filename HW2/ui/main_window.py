import sys
import os
import torch
import torch.nn as nn
from torchvision import models
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.font_manager as fm


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("影像處理系統")
        self.setGeometry(100, 100, 1200, 800)
        
        # 1. 設定設備，確保在使用 self.device 前已定義
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'使用設備: {self.device}')  # 可以幫助確認設備是否正確初始化
        
        # 2. 類別名稱對應 MNIST
        self.mnist_class_names = [str(i) for i in range(10)]
        
        # 3. 建立主要的 widget 和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左側控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 載入按鈕
        self.load_image_btn = QPushButton("Load Image")
        self.load_video_btn = QPushButton("Load Video")
        
        # 功能按鈕 - 第一組
        self.show_structure_btn = QPushButton("1.1 Show Structure")
        self.show_acc_loss_btn = QPushButton("1.2 Show Acc and Loss")
        self.predict_btn = QPushButton("1.3 Predict")
        
        # 預測標籤
        self.predict_label = QLabel("預測類別: N/A")
        
        # 功能按鈕 - 第二組
        self.q2_load_image_btn = QPushButton("Q2 Load Image")
        self.show_image_btn = QPushButton("2.1 Show Image")
        self.show_model_structure_btn = QPushButton("2.2 Show Model Structure")
        self.show_compression_btn = QPushButton("2.3 Show Compression")
        self.inference_btn = QPushButton("2.4 Inference")
        
        # 文字標籤
        self.text_label = QLabel("TextLabel")
        
        # 添加訓練按鈕
        self.train_btn = QPushButton("Train Model")
        
        # 狀態顯示標籤
        self.status_label = QLabel("Status: Ready")
        
        # 類別名稱對應
        self.class_names = ['Cat', 'Dog']
        
        # 添加所有控制元件到左側面板
        left_layout.addWidget(self.load_image_btn)
        left_layout.addWidget(self.load_video_btn)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.show_structure_btn)
        left_layout.addWidget(self.show_acc_loss_btn)
        left_layout.addWidget(self.predict_btn)
        left_layout.addWidget(self.predict_label)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.q2_load_image_btn)
        left_layout.addWidget(self.show_image_btn)
        left_layout.addWidget(self.show_model_structure_btn)
        left_layout.addWidget(self.show_compression_btn)
        left_layout.addWidget(self.inference_btn)
        left_layout.addWidget(self.text_label)
        left_layout.addWidget(self.train_btn)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch()
        
        # 右側顯示區域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 上下兩個顯示區域
        self.display_label_1 = QLabel()
        self.display_label_2 = QLabel()
        self.display_label_1.setStyleSheet("border: 1px solid black")
        self.display_label_2.setStyleSheet("border: 1px solid black")
        self.display_label_1.setMinimumSize(400, 300)
        self.display_label_2.setMinimumSize(400, 300)
        
        right_layout.addWidget(self.display_label_1)
        right_layout.addWidget(self.display_label_2)
        
        # 將左右面板添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # 初始化模型和數據集
        self.model_vgg16 = self.initialize_vgg16()
        self.model_resnet50_no_erasing = self.initialize_resnet50(feature_extract=False)
        self.model_resnet50_with_erasing = self.initialize_resnet50(feature_extract=False)
        # self.train_loader, self.test_loader = self.prepare_data()
        
        # 載入最佳模型
        self.load_model()
        
        # 連接按鈕信號
        self.setup_connections()
        
    def initialize_vgg16(self):
        """
        初始化 VGG16 模型，加入 Batch Normalization，並調整輸入與輸出層。

        Returns:
            model (torch.nn.Module): 初始化後的 VGG16 模型。
        """
        from torchvision.models import VGG16_BN_Weights

        # 使用預訓練權重，如果沒有預訓練權重則設為 None
        weights = VGG16_BN_Weights.DEFAULT

        model = models.vgg16_bn(weights=weights)  # 更新使用 'weights' 參數
        # 調整第一層以接受單通道圖片
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # 調整輸出層為 10 類（根據 MNIST）
        model.classifier[6] = nn.Linear(4096, 10)
        model = model.to(self.device)
        return model

    def initialize_resnet50(self, num_classes=2, feature_extract=True):
        """
        初始化 ResNet50 模型，並改最後的全連接層。

        Args:
            num_classes (int): 輸出類別數量。
            feature_extract (bool): 是否凍結除最後一層以外的所有層。

        Returns:
            model (torch.nn.Module): 初始化後的 ResNet50 模型。
        """
        model = models.resnet50(pretrained=True)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )
        return model

    # def prepare_data(self):
    #     transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406],
    #                              [0.229, 0.224, 0.225])
    #     ])
    #     train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    #     test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    #     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #     return train_loader, test_loader

    def load_model(self):
        """
        載入 VGG16 和 ResNet50 模型的權重。
        """
        # 載入 VGG16 模型權重
        model_path_vgg16 = "best_model_vgg16.pth"
        if os.path.exists(model_path_vgg16):
            try:
                self.model_vgg16.load_state_dict(torch.load(model_path_vgg16, map_location=self.device))
                self.status_label.setText("Status: VGG16 模型已載入。")
                print("VGG16 模型已成功載入。")
                # 打印第一層卷積層以確認輸入通道數
                print("VGG16 第一層卷積層:", self.model_vgg16.features[0])
            except Exception as e:
                self.status_label.setText("Status: VGG16 模型載入失敗。")
                QMessageBox.critical(self, "Error", f"VGG16 模型載入失敗: {e}")
        else:
            self.status_label.setText("Status: VGG16 模型未找到。")
            print("VGG16 模型檔案未找到。")
        
        # 載入 ResNet50 模型（無 Random Erasing）
        model_path_resnet_no_erasing = "resnet50_no_erasing.pth"
        if os.path.exists(model_path_resnet_no_erasing):
            try:
                self.model_resnet50_no_erasing.load_state_dict(torch.load(model_path_resnet_no_erasing, map_location=self.device))
                print("ResNet50 (無 Random Erasing) 模型已載入。")
            except Exception as e:
                print(f"ResNet50 (無 Random Erasing) 模型載入失敗: {e}")
        else:
            print("ResNet50 (無 Random Erasing) 模型檔案未找到。")
        
        # 載入 ResNet50 模型（使用 Random Erasing）
        model_path_resnet_with_erasing = "resnet50_with_erasing.pth"
        if os.path.exists(model_path_resnet_with_erasing):
            try:
                self.model_resnet50_with_erasing.load_state_dict(torch.load(model_path_resnet_with_erasing, map_location=self.device))
                print("ResNet50 (使用 Random Erasing) 模型已載入。")
            except Exception as e:
                print(f"ResNet50 (使用 Random Erasing) 模型載入失敗: {e}")
        else:
            print("ResNet50 (使用 Random Erasing) 模型檔案未找到。")

    def show_model_structure(self):
        from torchsummary import summary  # 確保已導入 summary
        print("VGG16 with Batch Normalization Model Structure:")
        try:
            summary(self.model_vgg16, (1, 32, 32))  # 單通道 32x32 圖片
        except Exception as e:
            QMessageBox.critical(self, "Error", f"顯示模型結構失敗: {e}")

    def show_acc_loss(self):
        # 顯示準確度和損失的圖表
        if os.path.exists("ui//accuracy.png") and os.path.exists("ui//loss.png"):
            acc_img = QPixmap("ui//accuracy.png")
            loss_img = QPixmap("ui//loss.png")
            self.display_label_1.setPixmap(acc_img)
            self.display_label_2.setPixmap(loss_img)
        else:
            QMessageBox.warning(self, "Warning", "Accuracy and loss images not found.")

    def predict(self):
        # 從指定路徑載入圖片並進行預測
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "選擇影像", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_name:
            # 使用 OpenCV 讀取影像並轉換為灰階
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (32, 32))  # 調整大小以符合模型輸入
            image = transforms.ToTensor()(image).unsqueeze(0)  # 轉換為 Tensor 並增加 batch 維度
            
            self.model_vgg16.eval()
            with torch.no_grad():
                outputs = self.model_vgg16(image)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # 顯示預測結果
            self.predict_label.setText(f"Predicted: {predicted[0].item()}")
            self.plot_probabilities(probabilities[0])

    def plot_probabilities(self, probabilities):
        plt.figure()
        plt.bar(range(10), probabilities.numpy(), color='blue')
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title('Predicted Probabilities')
        plt.xticks(range(10))
        plt.show()

    def setup_connections(self):
        """
        連接所有按鈕的信號與槽。
        """
        self.show_model_structure_btn.clicked.connect(self.show_resnet50_structure)
        self.show_acc_loss_btn.clicked.connect(self.show_acc_loss)
        self.show_compression_btn.clicked.connect(self.show_resnet50_comparison)
        self.predict_btn.clicked.connect(self.inference_vgg16)  # 連接 VGG16 預測
        self.train_btn.clicked.connect(self.start_training)
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_video_btn.clicked.connect(self.load_video)
        self.q2_load_image_btn.clicked.connect(self.q2_load_image)
        self.show_image_btn.clicked.connect(self.show_image)
        self.inference_btn.clicked.connect(self.inference_resnet50)
        self.show_structure_btn.clicked.connect(self.show_model_structure)

    def load_image(self):
        # 打開文件對話框以選擇影像文件
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "選擇影像", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_name:
            # 使用 OpenCV 讀取影像
            image = cv2.imread(file_name)
            # 將影像從 BGR 轉換為 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 將影像轉換為 QImage
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 在 QLabel 中顯示影像
            self.display_label_1.setPixmap(QPixmap.fromImage(q_image))

    def load_video(self):
        # 在這裡添加加載影片的邏輯
        pass 

    def q2_load_image(self):
        # 在這裡添加 Q2 加載影像的邏輯
        pass

    def show_image(self):
        # 在這裡顯示影像的邏輯
        pass

    def show_compression(self):
        # 在這裡顯示壓縮的邏輯
        pass


    def start_training(self):
        self.status_label.setText("Status: Training...")
        self.train_btn.setEnabled(False)  # 禁用訓練按鈕

        # 開始訓練模型
        try:
            self.train_acc, self.train_loss = train_model(self.model_vgg16, self.train_loader)
            plot_training_results(self.train_acc, self.train_loss)

            # 保存模型
            model_save_path = os.path.join("resources/models", "vgg16_mnist.pth")
            torch.save(self.model_vgg16.state_dict(), model_save_path)
            self.status_label.setText("Status: Training completed. Model saved.")
        except Exception as e:
            self.status_label.setText("Status: Training failed.")
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.train_btn.setEnabled(True)  # 啟用訓練按鈕

    def show_resnet50_structure(self):
        print("ResNet50 (無 Random Erasing) 模型架構:")
        summary(self.model_resnet50_no_erasing, input_size=(3, 224, 224))
        
        print("\nResNet50 (使用 Random Erasing) 模型架構:")
        summary(self.model_resnet50_with_erasing, input_size=(3, 224, 224))

    def show_resnet50_comparison(self):
        from PyQt5.QtWidgets import QLabel, QVBoxLayout, QDialog
        from PyQt5.QtGui import QPixmap
        
        # 檢查圖像路徑是否存在
        comparison_image_path = "com.png"
        if not os.path.exists(comparison_image_path):
            QMessageBox.warning(self, "Warning", "比較圖表未找到。")
            return
        
        # 創建新視窗
        dialog = QDialog(self)
        dialog.setWindowTitle("ResNet50 準確度比較")
        dialog.setGeometry(150, 150, 800, 600)
        
        # 加載圖片
        pixmap = QPixmap(comparison_image_path)
        label = QLabel(dialog)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        
        # 設置布局
        layout = QVBoxLayout()
        layout.addWidget(label)
        dialog.setLayout(layout)
        
        # 顯示視窗
        dialog.exec_()

    def inference_resnet50(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from PyQt5.QtGui import QImage, QPixmap
        from torchvision import transforms
        from PIL import Image
        import torch
        
        # 選擇圖片檔案
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "選擇要預測的圖片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )
        
        if file_name:
            try:
                # 使用 PIL 讀取圖片
                image = Image.open(file_name).convert("RGB")
                
                # 定義預處理轉換
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
                
                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)  # 增加 batch 維度
                
                # 選擇模型並設置設備
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model_resnet50_with_erasing.to(device)
                self.model_resnet50_with_erasing.eval()
                
                # 預測
                with torch.no_grad():
                    input_batch = input_batch.to(device)
                    output = self.model_resnet50_with_erasing(input_batch)
                    _, predicted = torch.max(output, 1)
                
                # 顯示預測結果於小視窗彈出
                class_idx = predicted.item()
                class_name = self.class_names[class_idx]
                
                QMessageBox.information(
                    self,
                    "預測結果",
                    f"預測類別: {class_name}"
                )
                
                # 顯示圖片
                # 將圖片轉換為可顯示格式
                image = image.resize((400, 400))
                image = image.convert("RGB")
                data = image.tobytes("raw", "RGB")
                q_image = QImage(data, image.size[0], image.size[1], QImage.Format_RGB888)
                self.display_label_1.setPixmap(QPixmap.fromImage(q_image))
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"預測失敗: {e}")

    def inference_vgg16(self):
        """
        使用 VGG16 模型進行圖片預測，並在同一視窗顯示預測類別及各類別機率值直方圖。
        """
        from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QLabel
        from PyQt5.QtGui import QImage, QPixmap
        from torchvision import transforms
        from PIL import Image
        import torch
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import os

        # 設定 matplotlib 支援中文
        font_path = os.path.join('ui', 'HanyiSentyPagoda Regular.ttf')
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            QMessageBox.warning(
                self,
                "警告",
                f"指定的字體檔案未找到: {font_path}\n將使用預設字體。"
            )

        plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

        # 選擇圖片檔案
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "選擇要預測的圖片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )

        if file_name:
            try:
                # 使用 PIL 讀取圖片並轉為灰階
                image = Image.open(file_name).convert("L")

                # 定義預處理轉換
                preprocess = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)  # 增加 batch 維度

                # 設置設備
                device = self.device
                self.model_vgg16.to(device)
                self.model_vgg16.eval()

                # 預測
                with torch.no_grad():
                    input_batch = input_batch.to(device)
                    output = self.model_vgg16(input_batch)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)

                # 取得預測結果
                class_idx = predicted.item()
                class_name = self.mnist_class_names[class_idx]
                probability_values = probabilities.cpu().numpy()[0]

                # 創建 QDialog 來顯示預測結果和直方圖
                dialog = QDialog(self)
                dialog.setWindowTitle("預測結果及機率分布")
                dialog.setGeometry(150, 150, 800, 600)

                layout = QVBoxLayout()

                # 預測類別標籤
                label = QLabel(f"預測類別: {class_name}")
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)

                # 創建 Matplotlib 圖表並嵌入到 FigureCanvas
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(range(len(probability_values)), probability_values, color='skyblue')
                ax.set_xlabel('類別')
                ax.set_ylabel('機率')
                ax.set_title(f'預測類別: {class_name} 的機率分布')
                ax.set_xticks(range(len(probability_values)))
                ax.set_xticklabels(self.mnist_class_names, rotation=45)
                ax.set_ylim([0, 1])

                # 在條形上顯示機率值
                for bar, prob in zip(bars, probability_values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{prob:.2%}', ha='center', va='bottom', fontsize=9)

                plt.tight_layout()

                # 嵌入圖表到 PyQt5
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)

                dialog.setLayout(layout)
                dialog.exec_()

                # 顯示圖片於 QLabel
                display_image = image.resize((400, 400))
                display_image = display_image.convert("L")  # 保持灰階
                q_image = QImage(display_image.tobytes("raw", "L"),
                                 display_image.size[0],
                                 display_image.size[1],
                                 QImage.Format_Grayscale8)
                self.display_label_1.setPixmap(QPixmap.fromImage(q_image))

            except Exception as e:
                QMessageBox.critical(self, "Error", f"預測失敗: {e}")