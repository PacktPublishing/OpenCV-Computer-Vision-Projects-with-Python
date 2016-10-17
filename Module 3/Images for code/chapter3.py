import cv2
import cv
import wx
import numpy as np
import time

class MyFrame(wx.Frame):
    def __init__(self, parent, id, title, capture, fps=15):
        # initialize screen capture
        self.capture = capture

        # determine window size and init wx.Frame
        ret,frame = self.capture.read()
        self.imgHeight,self.imgWidth = frame.shape[:2]
        buffer = np.zeros((self.imgWidth,self.imgHeight,3),np.uint8)
        self.bmp = wx.BitmapFromBuffer(self.imgWidth, self.imgHeight, buffer)
        wx.Frame.__init__(self, parent, id, title, size=(self.imgWidth, self.imgHeight))
        
        # counteract flicker
        def disable_event(*pargs,**kwargs):
            pass
        self.Bind(wx.EVT_ERASE_BACKGROUND, disable_event)

        self.CreateLayout()
        self.InitializeAlgorithm()

        # set up periodic screen capture
        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)
        self.Bind(wx.EVT_TIMER, self.NextFrame)


    def InitializeAlgorithm(self):
		# initialize SURF
        self.minHessian = 400
        self.SURF = cv2.SURF(self.minHessian)

		# template image: "train" image
		# later on compared ot each video frame: "query" image
        self.imgObj = cv2.imread("salinger.jpg", cv2.CV_8UC1)
        self.shTrain = self.imgObj.shape[:2]
        self.keyTrain,self.descTrain = self.SURF.detectAndCompute(self.imgObj,None)

		# initialize FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

		# initialize tracking
        self.lastHinv = np.zeros((3,3))
        self.framesWithoutSuccess = 0
        self.maxFramesWithoutSuccess = 5
        self.maxErrorConsecutiveHinv = 50.

    def CreateLayout(self):
        self.pnl = wx.Panel(self, -1, size=(self.imgWidth,self.imgHeight))
        self.pnl.SetBackgroundColour(wx.BLACK)

        self.SetMinSize((self.imgWidth, self.imgHeight))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.pnl, 1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        self.Centre()

    def NextFrame(self, event):
        # acquire new frame, ignore timestamp
        ret,frame = self.capture.read()
        if ret:
			# process frame, return success if object of interest found
            success,warpedFrame = self.ProcessFrame(frame)

            if success:
                # update buffer and paint
                self.bmp.CopyFromBuffer(warpedFrame)
                deviceContext = wx.BufferedPaintDC(self.pnl)
                deviceContext.DrawBitmap(self.bmp, 0, 0)
                del deviceContext


    def ProcessFrame(self, frame):
		""" processes each frame """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		# create a working copy (grayscale) of the frame
		# and store its shape for convenience
        imgQuery = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shQuery = imgQuery.shape[:2] # rows,cols

		# --- feature extraction
		# detect keypoints in the query image (video frame)
		# using SURF descriptor
		keyQuery,descQuery = self.ExtractFeatures(imgQuery)

		# --- feature matching
		# returns a list of good matches using FLANN
		# based on a scene and its feature descriptor
		goodMatches = self.MatchFeatures(descQuery)

		# early outlier detection and rejection
        if len(goodMatches)<4:
            self.framesWithoutSuccess=self.framesWithoutSuccess+1
            return [False,frame]

		# --- corner point detection
		# calculates the homography matrix needed to convert between
		# keypoints from the train image and the query image
		dstCorners = self.DetectCornerPoints(keyQuery, goodMatches)

		# early outlier detection and rejection
        # if any corners lie significantly outside the image, skip frame
        if np.any(filter(lambda x: x[0]<-20 or x[1]<-20 or x[0]>shQuery[1]+20 or x[1]>shQuery[0]+20, dstCorners)):
            self.framesWithoutSuccess=self.framesWithoutSuccess+1
            return [False,frame]

		# early outlier detection and rejection
		# find the area of the quadrilateral that the four corner points spans
        area = 0
        for i in xrange(0,4):
            iNext = (i+1)%4
            area = area + (dstCorners[i][0]*dstCorners[iNext][1]-dstCorners[i][1]*dstCorners[iNext][0])/2.

		# early outlier detection and rejection
		# reject corner points if area is unreasonable
        if area<np.prod(shQuery)/16. or area>np.prod(shQuery)/2.:
            self.framesWithoutSuccess=self.framesWithoutSuccess+1
            return [False,frame]

		# adjust x-coordinate (col) of corner points so that they can be drawn next
		# to the train image (add self.shTrain[1])
		dstCorners = [(np.int(dstCorners[i][0]+self.shTrain[1]),np.int(dstCorners[i][1])) for i in xrange(len(dstCorners))]

		# outline corner points of train image in query image
        imgFlann = drawGoodMatches(self.imgObj, self.keyTrain, imgQuery, keyQuery, goodMatches)
        for i in xrange(0,len(dstCorners)):
            cv2.line(imgFlann, dstCorners[i], dstCorners[(i+1)%4], (0,255,0), 3)

        cv2.imshow("imgFlann",imgFlann)

        # --- bring object of interest to frontoparallel plane
		Hinv = self.WarpKeypoints(goodMatches, keyQuery, shQuery)

		# outlier rejection
		# if last frame recent: new Hinv must be similar to last one
		# else: accept whatever Hinv is found at this point
        if self.framesWithoutSuccess < self.maxFramesWithoutSuccess:
            if np.linalg.norm(Hinv - self.lastHinv) > self.maxErrorConsecutiveHinv:
                self.framesWithoutSuccess=self.framesWithoutSuccess+1
                return [False,frame]

		# reset counters and update Hinv
        self.framesWithoutSuccess = 0
        self.lastH = Hinv

        imgOut = cv2.warpPerspective(imgQuery, Hinv, dstSz)
        imgOut = cv2.cvtColor(imgOut, cv2.COLOR_GRAY2RGB)
		
        return [True,imgOut]

		
	def ExtractFeatures(self, frame):
		""" detect keypoints using SURF descriptor
		    return [keypoints,descriptor] of the image features
		"""
		return self.SURF.detectAndCompute(frame, None)
		
	def MatchFeatures(self, descFrame):
		""" find matches between the descriptor of an input frame and the stored
		    train image
		"""
        # find 2 best matches (kNN with k=2)
        matches = self.flann.knnMatch(self.descTrain, descFrame, k=2)

        # discard bad matches, ratio test as per Lowe's paper
        goodMatches = filter(lambda x: x[0].distance<0.7*x[1].distance, matches)
        goodMatches = [goodMatches[i][0] for i in xrange(len(goodMatches))]

        return goodMatches
		
	def FindCornerPoints(self, keyFrame, goodMatches):
		""" find the homography matrix between train and query image,
			locate the corner points of the train image in the query image
			return list of corner points
		"""
        # find homography using RANSAC
        srcPoints = [self.keyTrain[goodMatches[i].trainIdx].pt for i in xrange(len(goodMatches))]
        dstPoints = [keyFrame[goodMatches[i].queryIdx].pt for i in xrange(len(goodMatches))]
        H,_ = cv2.findHomography(np.array(srcPoints), np.array(dstPoints), cv2.RANSAC)

        # outline train image in query image
        srcCorners = np.array([(0,0), (self.shTrain[1],0), (self.shTrain[1],self.shTrain[0]), (0,self.shTrain[0])], dtype=np.float32)
        dstCorners = cv2.perspectiveTransform(srcCorners[None,:,:], H)

		# convert to tuple
        dstCorners = map(tuple,dstCorners[0])
		return dstCorners

	def WarpKeypoints(self, goodMatches, keyFrame, shFrame):
		""" computes the homography matrix needed to bring the object in the
		    scene to the frontoparallel plane
		"""
		# bring object to frontoparallel plane: centered, up-right
        dstSz = (shFrame[1], shFrame[0]) # cols,rows
        scaleRow = 1./self.shTrain[0]*dstSz[1]/2.
        biasRow = dstSz[0]/4.
        scaleCol = 1./self.shTrain[1]*dstSz[0]*3/4.
        biasCol = dstSz[1]/8.

		# source points are the ones in the query image
        srcPoints = [keyFrame[goodMatches[i].queryIdx].pt for i in xrange(len(goodMatches))]
		
		# destination points are the ones in the train image
		# off-set in space so that the image is centered
        dstPoints = [self.keyTrain[goodMatches[i].trainIdx].pt for i in xrange(len(goodMatches))]
        dstPoints = map(lambda x: (x[0]*scaleRow+biasRow, x[1]*scaleCol+biasCol), dstPoints)
		
		# find homography
        Hinv,_ = cv2.findHomography(np.array(srcPoints), np.array(dstPoints), cv2.RANSAC)

		return Hinv


def drawGoodMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for m in matches:
        # Get the matching keypoints for each of the images

        (c1,r1) = kp1[m.trainIdx].pt
        (c2,r2) = kp2[m.queryIdx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(c1),int(r1)), 4, (255,0,0), 1)   
        cv2.circle(out, (int(c2)+cols1,int(r2)), 4, (255,0,0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(c1),int(r1)), (int(c2)+cols1,int(r2)), (255,0,0), 1)

    return out


def main():
    capture = cv2.VideoCapture(0)
    if not(capture.isOpened()):
        capture.open()

    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    app = wx.App()
    frame = MyFrame(None, -1, 'chapter3.py', capture, fps=10)
    frame.Show(True)
    app.MainLoop()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()