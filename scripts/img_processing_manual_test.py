import matplotlib.pyplot as plt

from src.config import config
from src.image_processing import preprocess_image

img = preprocess_image("tests/test_images/test_menu.jpg")
plt.imshow(img, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()  # Display the image


