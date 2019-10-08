from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, width, height):
        self.height = height
        self.width = width

    def preprocess(self, image):
        return extract_patches_2d(image=image, patch_size=(self.height, self.width), max_patches=1)[0]