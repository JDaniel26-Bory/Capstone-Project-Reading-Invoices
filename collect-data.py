# Libraries
import os
import cv2
from PIL import Image
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from time import time

imgpath = 'data/imagenes-facturas1'

# List
images = []
clases = []
lista = os.listdir(imgpath)

# Leemos las imagenes del DB
for lis in lista:
    # Leemos las imagenes de los rostros
    imgdb = cv2.imread(f'{imgpath}/{lis}')
    # Almacenamos imagen
    images.append(imgdb)
    # Almacenamos nombre
    clases.append(os.path.splitext(lis)[0])

count = 0

numImganes = len(images)

print(f'Numero imagenes: {numImganes}')

outFolderPath = 'data/debug/'

classID = 0

home = os.getcwd()
# Config Path
config_path = os.path.join(home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# CheckPoint Weights
check_point_path = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'

# Model
model = load_model(config_path, check_point_path)

# Prompt
text_prompt = 'invoice'
box_threshold = 0.35
text_threshold = 0.25

save = True

while(count < numImganes):
    img = images[count]
    imgCopy = img.copy()

    # Transform
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Convert img to PIL object
    img_source = Image.fromarray(img).convert("RGB")

    # Convert PIL image onject to transform object
    img_transform, _ = transform(img_source, None)

    # Predict
    boxes, logits, phrases = predict(
        model=model,
        image=img_transform,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device='cpu')
    
    infoList = []
    

    if len(boxes) != 0:
        alto, ancho, c = img.shape
        info = boxes[0][0]
        xc, yc, an, al = boxes[0][0],boxes[0][1],boxes[0][2],boxes[0][3]

        # Error < 0
        if xc < 0: xc = 0
        if yc < 0: yc = 0
        if an < 0: an = 0
        if al < 0: al = 0
        # Error > 1
        if xc > 1: xc = 1
        if yc > 1: yc = 1
        if an > 1: an = 1
        if al > 1: al = 1

        infoList.append(f'{classID} {xc} {yc} {an} {al}')

        if save:
            # Name
            timeNow = time()
            timeNow = str(timeNow)
            timeNow = timeNow.split('.')
            timeNow = timeNow[0] + timeNow[1]
        
            # Save Image Without Draw
            cv2.imwrite(f"{outFolderPath}/{timeNow}.jpg", imgCopy)
            # Save Text
            for info in infoList:
                f = open(f"{outFolderPath}/{timeNow}.txt", 'a')
                f.write(info)
                f.close()

        



    # Annotated
    annotated_img = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)

    # display the output
    out_frame = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)


    cv2.imshow('DINO', out_frame) 
    count += 1

    t = cv2.waitKey(0)
    if t == 27:
        break

cv2.destroyAllWindows()
