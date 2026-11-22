from leafapp.labels import format_class_name


def test_format_class_name():
    """Dataset labels are shown as "Plant - Disease"."""
    assert format_class_name("Tomato___Leaf_Mold") == "Tomato - Leaf Mold"
    assert format_class_name("Apple___healthy") == "Apple - healthy"


def test_format_class_name_drops_trailing_underscore():
    """Some labels end in an underscore, which should not become a trailing space."""
    assert format_class_name("Corn_(maize)___Common_rust_") == "Corn (maize) - Common rust"
