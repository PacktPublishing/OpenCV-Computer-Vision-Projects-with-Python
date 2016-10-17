import cv2
import freenect
import wx
import numpy as np
import time

class MyFrame(wx.Frame):
    def __init__(self, parent, id, title, capture, fps=15):
        # initialize screen capture
        self.capture = capture

        # determine window size and init wx.Frame
        frame,_ = freenect.sync_get_depth()
        self.imgHeight,self.imgWidth = frame.shape[:2]
        buffer = np.zeros((self.imgWidth,self.imgHeight,3),np.uint8)
        self.bmp = wx.BitmapFromBuffer(self.imgWidth, self.imgHeight, buffer)
        wx.Frame.__init__(self, parent, id, title, size=(self.imgWidth, self.imgHeight))

        # set up periodic screen capture
        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)
        self.Bind(wx.EVT_TIMER, self.NextFrame)
        
        # counteract flicker
        def disable_event(*pargs,**kwargs):
            pass
        self.Bind(wx.EVT_ERASE_BACKGROUND, disable_event)

        # create the layout, which draws all buttons and
        # connects events to class methods
        self.CreateLayout()


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
        frame,_ = freenect.sync_get_depth()

        # clip max depth to 1023, convert to 8-bit grayscale
        np.clip(frame, 0, 2**10 - 1, frame)
        frame >>= 2
        frame = frame.astype(np.uint8)

        # segment hand, detect number of fingers, return
        # annotated RGB image
        frame = self.ProcessFrame(frame)

        # update buffer and paint
        self.bmp.CopyFromBuffer(frame)
        deviceContext = wx.BufferedPaintDC(self.pnl)
        deviceContext.DrawBitmap(self.bmp, 0, 0)
        del deviceContext


    def ProcessFrame(self, frame):
        # segment arm region
        segment = self.SegmentArm(frame)

        # make a copy of the segmented image to draw on
        draw = cv2.cvtColor(segment, cv2.COLOR_GRAY2RGB)

        # draw some helpers for correctly placing hand
        cv2.circle(draw,(self.imgWidth/2,self.imgHeight/2),3,[255,102,0],2)       
        cv2.rectangle(draw, (self.imgWidth/3,self.imgHeight/3), (self.imgWidth*2/3, self.imgHeight*2/3), [255,102,0],2)

        # find the hull of the segmented area, and based on that find the
        # convexity defects
        [contours,defects] = self.FindHullDefects(segment)

        # detect the number of fingers depending on the contours and convexity defects
        # draw defects that belong to fingers green, others red
        [nofingers,draw] = self.DetectNumberFingers(contours, defects, draw)

        # print number of fingers on image
        cv2.putText(draw, str(nofingers), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        return draw

    def SegmentArm(self, frame):
        """ segments the arm region based on depth """
        # find center (21x21 pixel) region of image frame
        centerHalf = 10 # half-width of 21 is 21/2-1
        center = frame[self.imgHeight/2-centerHalf:self.imgHeight/2+centerHalf,
            self.imgWidth/2-centerHalf:self.imgWidth/2+centerHalf]

        # find median depth value of center region
        center = np.reshape(center, np.prod(center.shape))
        medVal = np.median( np.reshape(center, np.prod(center.shape)) )

        # try this instead:
        absDepthDev = 14
        frame = np.where(abs(frame-medVal) <= absDepthDev, 128, 0).astype(np.uint8)

        # morphological
        kernel = np.ones((3,3), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

        # connected component
        smallKernel = 3
        frame[self.imgHeight/2-smallKernel:self.imgHeight/2+smallKernel,
            self.imgWidth/2-smallKernel:self.imgWidth/2+smallKernel] = 128

        mask = np.zeros((self.imgHeight+2,self.imgWidth+2), np.uint8)
        flood = frame.copy()
        cv2.floodFill(flood, mask, (self.imgWidth/2,self.imgHeight/2), 255, flags=4|(255<<8))

        ret,flooded = cv2.threshold(flood, 129, 255, cv2.THRESH_BINARY)

        return flooded


    def FindHullDefects(self, segment):
        _,contours,hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find largest area contour
        max_area = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area>max_area:
                cnt = contours[i]
                max_area = area

        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        return [cnt,defects]


    def DetectNumberFingers(self, contours, defects, draw):
        """ determines the number of extended fingers based on a contour and convexity defects """

        # cut-off angle (deg): everything below this is a convexity point that belongs to two
        # extended fingers
        angleFingerDeg = 80.0

        # if there are no convexity defects, possibly no hull found or no fingers extended
        if defects is None:
            return [0,draw]

        # we assume the wrist will generate two convexity defects (one on each side), so if
        # there are no additional defect points, there are no fingers extended
        if len(defects) <= 2:
            return [0,draw]

        # if there is a sufficient amount of convexity defects, we will find a defect point
        # between two fingers so to get the number of fingers, start counting at 1
        nofingers = 1

        for i in range(defects.shape[0]):
            # each defect point is a 4-tuple
            s,e,f,d = defects[i,0]
            start = tuple(contours[s][0])
            end = tuple(contours[e][0])
            far = tuple(contours[f][0])

            # draw the hull
            cv2.line(draw,start,end,[0,255,0],2)

            # if angle is below a threshold, defect point belongs to two extended fingers
            if angle(np.subtract(start,far), np.subtract(end,far)) < angleFingerDeg/180.0*np.pi:
                # increment number of fingers
                nofingers = nofingers + 1

                # draw point as green
                cv2.circle(draw,far,5,[0,255,0],-1)
            else:
                # draw point as red
                cv2.circle(draw,far,5,[255,0,0],-1)

        # make sure we cap the number of fingers
        return [min(5, nofingers),draw]


def angle(v1, v2):
    """ returns the angle (in radians) between two array-like vectors using the
        cross-product method, which is more accurate for small angles than the
        dot-product-acos method."""
    return np.arctan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))


def main():
    device = cv2.CAP_OPENNI
    capture = cv2.VideoCapture(device)
    if not(capture.isOpened()):
        capture.open(device)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = wx.App()
    frame = MyFrame(None, -1, 'chapter2.py', capture)
    frame.Show(True)
#   self.SetTopWindow(frame)
    app.MainLoop()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()