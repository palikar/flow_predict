
import numpy as np
import PIL

from PIL import Image

def main():
    print('Gen mask')

    mask_img = np.ones((290, 1567))

    mask_img[96+17:168+18, 113-17:186-18] = 0

    mask_img = Image.fromarray(np.uint8(mask_img))

    mask_img.save("./mask.png")


    # print(img.shape)
    # print(img)

    # flow = Image.open("/home/arnaud/temp/flow_predict/project/data/generated_data/c_1/images_x/flow.0096.png").convert("L")
    # mask = Image.open("./mask.png")

    # flow = flow.resize((1024, 256), Image.BICUBIC)
    # mask = mask.resize((1024, 256), Image.BICUBIC)

    # flow = np.array(flow)
    # mask = np.array(mask)
 
    # print(flow.shape)
    # print(mask.shape)

    # mult = flow*mask
    # im = Image.fromarray(np.uint8(mult))
    # im.show()
    
    # flow.show()
    # im = Image.fromarray(np.uint8(img))
    # im.save("./mask.png")


if __name__ == '__main__':
    main()
