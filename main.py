from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN


def detect_face():
    # plot photo with detected faces using opencv cascade classifier
    import cv2
    from cv2 import imread
    from cv2 import imshow
    from cv2 import waitKey
    from cv2 import destroyAllWindows
    from cv2 import CascadeClassifier
    from cv2 import rectangle
    # load the photograph
    pixels = imread('test6.jpg')
    # load the pre-trained model
    classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
    # perform face detection
    # bboxes = classifier.detectMultiScale(pixels)
    bboxes = classifier.detectMultiScale(pixels, 1.1, 5)
    # print bounding box for each detected face
    for box in bboxes:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(pixels, (x, y), (x2, y2), (0, 255, 255), 3)
    # show the image
    pixels = cv2.resize(pixels, (960, 540))
    imshow('face detection', pixels)
    # keep the window open until we press a key
    waitKey(0)
    # close the window
    destroyAllWindows()


# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    print('confidence')
    for face in result_list:
        print(face['confidence'])
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='yellow')
        # draw the box
        ax.add_patch(rect)
    # show the plot
    pyplot.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detect_face()
    filename = 'test6.jpg'
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    draw_image_with_boxes(filename, faces)

