from PIL import Image
import io

def ex():
    with open("C:/Users/Tim/PycharmProjects/reinforcement-learning/im.png", "rb") as file:
        im = file.read()
        b = bytearray(im)
        return b

def main():
    #b = ex()
    with open("bytes.txt", "r") as file:
        b = list(file.read())
        print(b)
        image = Image.open(io.BytesIO(b))
        image.show()

main()