import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img,vertices):
    mask=np.zeros_like(img)
    #channel_count=img.shape[2]
    match_mask_color=255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_img=cv2.bitwise_and(img,mask)
    return masked_img

def draw_lines(img,lines):
    img=np.copy(img)
    blank_img=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)


    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_img,(x1,y1),(x2,y2),(0,255,0),2)

    img=cv2.addWeighted(img,0.8,blank_img,1,0.0)
    return img

'''image=cv2.imread('road.png')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)'''
def process(image):
    print(image.shape)
    height=image.shape[0]
    width=image.shape[1]

    region_of_interest_vertices=[ (0,height),
        (width/2,height/2),
        (width,height)]
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny=cv2.Canny(grey,100,200)
    cropped_image= region_of_interest(canny,
                                     np.array([region_of_interest_vertices],np.int32),)
    lines=cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi/60,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=25)

    image_with_line=draw_lines(image,lines)
    return image_with_line

cap=cv2.VideoCapture('road.mp4')

while(cap.isOpened()):
    ret,frame=cap.read()
    frame=process(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()