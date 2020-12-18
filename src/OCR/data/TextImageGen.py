from PIL import Image

from src.OCR.data.Fonts_and_Languages import gen_random_character_batches
from src.OCR.data.Text import Text

if __name__ == '__main__':
    img = Image.new('RGB', (1100, 128), (100, 50, 150))
    text_gen = gen_random_character_batches(img)
    next_text = next(text_gen)
    text = Text(next_text)
    text.draw_to_image_centered(img, (255, 255, 255))

    # img
