import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np
import pandas as pd


class CoFruit(QMainWindow):
    def __init__(self):
        super(CoFruit, self).__init__()
        loadUi("ui/main.ui", self)
        self.setWindowTitle("CoFruit")
        self.open_img.clicked.connect(self.OpenImg)
        self.analys.clicked.connect(self.Analys)
        self.clear.clicked.connect(self.Reset)
        self.save.clicked.connect(self.Save)

        self.rgb_img = None  # Initialize as None

        self.hmin.setMinimum(0)
        self.hmin.setMaximum(179)
        self.smin.setMinimum(0)
        self.smin.setMaximum(255)
        self.vmin.setMinimum(0)
        self.vmin.setMaximum(255)
        self.hmax.setMinimum(0)
        self.hmax.setMaximum(179)
        self.smax.setMinimum(0)
        self.smax.setMaximum(255)
        self.vmax.setMinimum(0)
        self.vmax.setMaximum(255)

        self.hmin.valueChanged.connect(self.updateThresholdedImage)
        self.smin.valueChanged.connect(self.updateThresholdedImage)
        self.vmin.valueChanged.connect(self.updateThresholdedImage)
        self.hmax.valueChanged.connect(self.updateThresholdedImage)
        self.smax.valueChanged.connect(self.updateThresholdedImage)
        self.vmax.valueChanged.connect(self.updateThresholdedImage)

    def OpenImg(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "","Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.filename = filename
        if filename:
            RawImg = cv2.imread(filename)
            if RawImg is not None:
                self.rgb_img = cv2.cvtColor(RawImg, cv2.COLOR_BGR2RGB)
                height, width, channel = RawImg.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.rgb_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                qImg = qImg.scaled(744, 480)
                image = QPixmap(qImg)
                self.raw_img.setPixmap(image)
                self.updateThresholdedImage()

    def updateThresholdedImage(self):
        if self.rgb_img is None:
            print("No image loaded.")
            return

        # Get threshold values from sliders
        h_min = self.hmin.value()
        s_min = self.smin.value()
        v_min = self.vmin.value()
        h_max = self.hmax.value()
        s_max = self.smax.value()
        v_max = self.vmax.value()

        # Create mask
        hsv_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, (h_min, s_min, v_min), (119, 255, 255))
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
        opening_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)
        closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel)
        blurred_mask = cv2.GaussianBlur(closing_mask, (5, 5), 0)
        mask = blurred_mask

        # Apply mask to RGB image
        masked_rgb = cv2.bitwise_and(self.rgb_img, self.rgb_img, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables for the centroid, width, and height
        cX, cY, width, height = 0, 0, 0, 0

        # If contours are found, process the largest one
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the moments to calculate the centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            # Get the bounding box of the largest contour
            x, y, width, height = cv2.boundingRect(largest_contour)

        # Output the mayor and minor diameters
        self.Mayor = height
        self.Minor = width

        # Convert mask to QImage for display
        h, w = mask.shape
        qImg_thresholded = QImage(mask.data, w, h, w, QImage.Format_Grayscale8)
        pixmap_thresholded = QPixmap.fromImage(qImg_thresholded)
        self.result_img.setPixmap(pixmap_thresholded)

        # Save masked RGB image and number of non-zero pixels for analysis
        self.masked_rgb = masked_rgb
        self.masked_hsv = mask
        self.num_pixels = cv2.countNonZero(mask)
        self.GetRGB(masked_rgb)

    def GetRGB(self, img):
        pixel = self.num_pixels
        color_row = np.sum(img, axis=0)
        sum_color = np.sum(color_row, axis=0)
        total_color = sum_color / pixel
        return self.CalculateRGB(total_color)

    def CalculateRGB(self, raw_data):
        b = raw_data[0]
        g = raw_data[1]
        r = raw_data[2]
        self.blue = int(b)
        self.green = int(g)
        self.red = int(r)

        return self.blue, self.green, self.red

    def CalculateHSI(self, R, G, B):
        R = R / 255.0
        G = G / 255.0
        B = B / 255.0

        I = (R + G + B) / 3.0

        min_RGB = min(R, G, B)
        if R + G + B == 0:
            S = 0
        else:
            S = 1 - (3 * min_RGB / (R + G + B))

        numerator = 0.5 * ((R - G) + (R - B))
        denominator = np.sqrt((R - G)**2 + (R - B) * (G - B))
        if denominator == 0:
            H = 0
        else:
            theta = np.arccos(numerator / denominator)
            if B <= G:
                H = theta
            else:
                H = 2 * np.pi - theta

        H = H * 180 / np.pi

        return H, S, I

    def Analys(self):
        if self.masked_rgb is None:
            print("No masked image available.")
            return

        self.Red, self.Green, self.Blue = self.GetRGB(self.masked_rgb)
        self.H, self.S, self.I = self.CalculateHSI(self.Red, self.Green, self.Blue)

        # Convert the masked RGB image to QImage for display
        h, w, ch = self.masked_rgb.shape
        #print(f"h: {h}, w: {w}, ch: {ch}")
        bytes_per_line = ch * w
        qImg_RGB = QImage(self.masked_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.Area = (self.num_pixels * 0.0013) + 65.994
        self.Weight = (self.num_pixels * 0.0028) - 14.459
        self.DMayor = (0.0186*self.Mayor) + 1.3044
        self.DMinor = (0.0128*self.Minor) + 3.4061



        self.pixmap_RGB = QPixmap.fromImage(qImg_RGB)
        self.raw_img.setPixmap(self.pixmap_RGB)
        self.pixel.setText(str(self.num_pixels))
        self.pixel_2.setText(str(self.Red))
        self.pixel_3.setText(str(self.Green))
        self.pixel_4.setText(str(self.Blue))
        self.hue.setText(f"{self.H:.2f}\u00b0")
        self.sat.setText(f"{100*self.S:.2f}%")
        self.inten.setText(f"{100*self.I:.2f}%")
        self.mayor.setText(str(round(self.DMayor,2)))
        self.minor.setText(str(round(self.DMinor,2)))
        self.weight.setText(str(round(self.Weight,2)))
        self.zarea.setText(f"{self.Area:.2f}")



    def Save(self):
        if self.masked_rgb is None or self.masked_rgb is None:
            print("No masked image available.")
            return
        save_dir = os.path.join(os.getcwd(), "result")
        filename = os.path.basename(self.filename)
        filename = os.path.splitext(filename)[0]
        masked_rgb_save_path = os.path.join(save_dir, f"rgb_{filename}.png")
        cv2.imwrite(masked_rgb_save_path, cv2.cvtColor(self.masked_rgb, cv2.COLOR_RGB2BGR))

        masked_hsv_save_path = os.path.join(save_dir, f"hsv_{filename}.png")
        cv2.imwrite(masked_hsv_save_path, cv2.cvtColor(self.masked_hsv, cv2.COLOR_RGB2BGR))
        print("Successfully saved masked image.")

        data = {
            "Filename": [filename],
            "Number of Pixels": [self.num_pixels],
            "Red": [self.Red],
            "Green": [self.Green],
            "Blue": [self.Blue],
            "Hue": [self.H],
            "Saturation": [self.S],
            "Intensity": [self.I],
            "Major Diameter": [self.Mayor],
            "Minor Diameter": [self.Minor]
        }

        df = pd.DataFrame(data)
        csv_save_path = os.path.join(save_dir, "result.csv")

        # Append the data to the CSV file
        if os.path.exists(csv_save_path):
            df.to_csv(csv_save_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_save_path, mode='w', header=True, index=False)

        print("Successfully saved data to CSV file.")

    def Reset(self):
        self.result_img.clear()
        self.raw_img.clear()
        self.pixel.setText("null")
        self.rgb_img = None  # Reset the image
        self.pixel_2.setText("null")
        self.pixel_3.setText("null")
        self.pixel_4.setText("null")
        self.hue.setText("null")
        self.sat.setText("null")
        self.inten.setText("null")
        self.mayor.setText("null")
        self.minor.setText("null")
        self.weight.setText("null")
        self.zarea.setText("null")
        print("Completed Reset")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = CoFruit()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(1920)
    widget.setFixedHeight(1080)
    widget.show()
    sys.exit(app.exec_())
