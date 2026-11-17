from PIL import Image

from leafapp.predict import preprocess_image


def test_preprocess_image():
    """A grayscale image becomes a [1, 3, 32, 32] array."""
    image = Image.new("L", (100, 80))

    x = preprocess_image(image, 32)

    assert x.shape == (1, 3, 32, 32)
