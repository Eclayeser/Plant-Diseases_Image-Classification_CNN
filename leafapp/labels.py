def format_class_name(name):
    """Turn a label such as "Tomato___Leaf_Mold" into "Tomato - Leaf Mold"."""
    return name.replace("___", " - ").replace("_", " ").strip()
