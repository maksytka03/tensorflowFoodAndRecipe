from train_model import class_names
from get_recipes import get_recipe, get_food_titles
from style import Style
import re
import os
import sys
import tensorflow as tf
from PIL import Image
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QGridLayout,
    QFileDialog,
    QTabWidget,
    QComboBox,
    QScrollArea,
    QVBoxLayout,
)
from PyQt6.QtGui import QPixmap, QPalette, QColor, QFont


# Class for scrollable label
class ScrollLabel(QScrollArea):
    # constructor
    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)

        # making widget resizable
        self.setWidgetResizable(True)

        # making qwidget object
        content = QWidget(self)
        self.setWidget(content)

        # vertical box layout
        lay = QVBoxLayout(content)

        # creating label
        self.label = QLabel(content)

        # setting alignment to the text
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # making label multi-line
        self.label.setWordWrap(True)

        # adding label to the layout
        lay.addWidget(self.label)

    # the setText method
    def setText(self, text):
        # setting text to the label
        self.label.setText(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Food Classifier")
        self.resize(1000, 500)

        layout = QGridLayout()
        central_widget = QWidget()

        # tab = QTabWidget(self, tabShape=QTabWidget.TabShape.Triangular)

        # Image classification
        # img_classification_page = QWidget(self)
        # img_classification_layout = QGridLayout()

        start_text = QLabel(
            "Hi there! Select an image of food and I will tell you what it is."
        )
        layout.addWidget(start_text, 0, 0, 1, 1)

        self.image = QLabel("")
        layout.addWidget(self.image, 1, 0, 1, 1)

        select_button = QPushButton("Select Image")
        select_button.clicked.connect(self.get_image)
        select_button.setStyleSheet(Style.button)
        layout.addWidget(select_button, 0, 1, 1, 1)

        self.predict_button = QPushButton("")
        self.predict_button.setVisible(False)
        self.predict_button.setStyleSheet(Style.button)
        self.predict_button.clicked.connect(
            lambda: self.predict_on_image(model_filename="model")
        )
        layout.addWidget(self.predict_button, 3, 0, 1, 1)

        self.recipe_button = QPushButton("")
        self.recipe_button.setVisible(False)
        self.recipe_button.setStyleSheet(Style.button)
        self.recipe_button.clicked.connect(self.show_recipe)
        layout.addWidget(self.recipe_button, 3, 1, 1, 1)

        self.image_label = QLabel("")
        self.image_label.setVisible(False)
        layout.addWidget(self.image_label, 1, 0, 1, 1)

        self.food_label = QLabel("")
        layout.addWidget(self.food_label, 2, 0, 1, 1)

        self.recipe_label = ScrollLabel(self)
        self.recipe_label.setVisible(False)
        layout.addWidget(self.recipe_label, 1, 1, 1, 1)

        self.recipe_dropdown = QComboBox()
        self.recipe_dropdown.setVisible(False)
        layout.addWidget(self.recipe_dropdown, 2, 1, 1, 1)

        # img_classification_page.setLayout(img_classification_layout)
        # tab.addTab(img_classification_page, "Image Classification")

        # # Forecasting
        # forecasting_page = QWidget(self)
        # forecasting_layout = QGridLayout()
        # text_2 = QLabel("Page 2")
        # forecasting_layout.addWidget(text_2)
        # forecasting_page.setLayout(forecasting_layout)
        # tab.addTab(forecasting_page, "Future Forecasting")

        # # Regression
        # regression_page = QWidget(tab)
        # regression_layout = QGridLayout()
        # text_3 = QLabel("Page 3")
        # regression_layout.addWidget(text_3)
        # regression_page.setLayout(regression_layout)
        # tab.addTab(regression_page, "Regression")

        # layout.addWidget(tab)

        self.setCentralWidget(central_widget)
        self.centralWidget().setLayout(layout)

        self.setStyleSheet(Style.background)

    def get_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open File", "c:\\", "Image files (*.png *.jpg)"
        )
        if filename:
            self.image.setPixmap(
                QPixmap(filename).scaled(self.width() / 2, self.width() / 2)
            )

            self.predict_button.setVisible(True)
            self.predict_button.setText("Predict")

            self.image_label.setText(filename)

    def predict_on_image(
        self,
        model_filename,
    ):  # TODO: Implement model choosing
        model = tf.keras.models.load_model(model_filename)
        image_filename = self.image_label.text()
        img = Image.open(image_filename)
        img = img.resize((224, 224))
        img_array = np.array(img)  # / 255 # no need to scale for EfficientNet
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]
        pred_class = np.argmax(pred)
        pred_prob = pred[pred_class]
        confidence = pred_prob * 100

        classes = class_names
        pred_class_name = classes[pred_class]

        self.pred_label = " ".join(pred_class_name.split("_")).title()
        self.food_label.setText(
            f"Food: {self.pred_label}, Confidence: {confidence:.1f}%"
        )

        self.recipe_label.setText("")  # clear
        self.recipe_label.setVisible(False)  # reset visibility

        self.recipe_dropdown.clear()  # clear
        self.recipe_dropdown.setVisible(True)
        self.recipe_dropdown.addItems(get_food_titles(self.pred_label.lower()))

        self.recipe_button.setVisible(True)
        self.recipe_button.setText("Show recipe")

    def show_recipe(self):
        food_name = self.pred_label.lower()
        food_recipe_name = self.recipe_dropdown.currentText()

        self.recipe_label.setVisible(True)

        ingredients, instructions = get_recipe(food_name, food_recipe_name)
        self.recipe_label.setText(
            f"Showing recipe for: {food_recipe_name}\n\nIngredients:\n{ingredients}\n\nInstructions: {instructions}"
        )

        # TODO: Redesign


if __name__ == "__main__":
    QApplication.setFont(QFont("Noto Sans", 11))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
