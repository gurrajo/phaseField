import cv2
vidcap = cv2.VideoCapture('animation_A11_k3.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames/_A11_k3frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1