import time
import wx
import cv2 as cv3
import numpy as np
from scipy.interpolate import UnivariateSpline

class MyFrame(wx.Frame):
	def __init__(self, parent, id, title, capture, fps=15):
		# initialize screen capture
		self.capture = capture
		ret,frame = self.capture.read()

		# determine window size and init wx.Frame
		self.imgHeight,self.imgWidth = frame.shape[:2]
		self.bmp = wx.BitmapFromBuffer(self.imgWidth, self.imgHeight, frame)
		wx.Frame.__init__(self, parent, id, title, size=(self.imgWidth, self.imgHeight+20))

		# set up periodic screen capture
		self.timer = wx.Timer(self)
		self.timer.Start(1000./fps)
		self.Bind(wx.EVT_TIMER, self.NextFrame)
		self.Bind(wx.EVT_PAINT, self.OnPaint)
		
		# counteract flicker
		def disable_event(*pargs,**kwargs):
			pass
		self.Bind(wx.EVT_ERASE_BACKGROUND, disable_event)

		# initialize image filters
		self.pencilSketchInit()
		self.colorFilterInit()
		self.cartoonizerInit()

		# create the layout, which draws all buttons and
		# connects events to class methods
		self.createLayout()


	def createLayout(self):
		self.pnl1 = wx.Panel(self, -1, size=(self.imgWidth,self.imgHeight))
		self.pnl1.SetBackgroundColour(wx.BLACK)

		# create a horizontal layout with all filter modes
		pnl2 = wx.Panel(self, -1 )
		self.mode_warm = wx.RadioButton(pnl2, -1, 'Warming Filter', (10, 10), style=wx.RB_GROUP)
		self.mode_cool = wx.RadioButton(pnl2, -1, 'Cooling Filter', (10, 10))
		self.mode_sketch = wx.RadioButton(pnl2, -1, 'Pencil Sketch', (10, 10))
		self.mode_cartoon = wx.RadioButton(pnl2, -1, 'Cartoon', (10, 10))
		hbox = wx.BoxSizer(wx.HORIZONTAL)
		hbox.Add(self.mode_warm, 1)
		hbox.Add(self.mode_cool, 1)
		hbox.Add(self.mode_sketch, 1)
		hbox.Add(self.mode_cartoon, 1)
		pnl2.SetSizer(hbox)

		# display the button layout beneath the video stream
		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.pnl1, 1, flag=wx.EXPAND)
		sizer.Add(pnl2, flag=wx.EXPAND | wx.BOTTOM | wx.TOP, border=1)

		self.SetMinSize((self.imgWidth, self.imgHeight))
		self.SetSizer(sizer)
		self.Centre()

	def OnPaint(self, evt):
		# read and draw buffered bitmap
		deviceContext = wx.BufferedPaintDC(self.pnl1)
		deviceContext.DrawBitmap(self.bmp, 0, 0)
		del deviceContext

	def NextFrame(self, event):
		ret, frame = self.capture.read()
		if ret:
			frame = cv3.cvtColor(frame, cv3.COLOR_BGR2RGB)

			if self.mode_warm.GetValue()==True:
				frame = self.colorFilterWarm(frame)
			elif self.mode_cool.GetValue()==True:
				frame = self.colorFilterCool(frame)
			elif self.mode_sketch.GetValue()==True:
				frame = self.pencilSketch(frame)
			elif self.mode_cartoon.GetValue()==True:
				frame = self.cartoonizer(frame)

			# update buffer and paint (EVT_PAINT triggered by Refresh)
			self.bmp.CopyFromBuffer(frame)
			self.Refresh(eraseBackground=False)

	def createLUT_8UC1(self, x, y):
	    spl = UnivariateSpline(x, y)
	    return spl(xrange(256))

	def pencilSketchInit(self):
		self.pencilsketch_bg = cv3.imread("pencilsketch_bg.jpg", cv3.CV_8UC1)
		self.pencilsketch_bg = cv3.resize(self.pencilsketch_bg, (self.imgWidth,self.imgHeight))

	def pencilSketch(self, imgRGB):
	    imgGray = cv3.cvtColor(imgRGB, cv3.COLOR_RGB2GRAY)
	    imgBlur = cv3.GaussianBlur(imgGray, (21,21), 0, 0)
	    imgBlend = cv3.divide(imgGray, imgBlur, scale=256)
	    imgBlend = cv3.multiply(imgBlend, self.pencilsketch_bg, scale=1./256)
	    return cv3.cvtColor(imgBlend, cv3.COLOR_GRAY2RGB)

	def colorFilterInit(self):
		# create look-up tables for increasing and decreasing a channel
		self.incrChLUT = self.createLUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
		self.decrChLUT = self.createLUT_8UC1([0, 64, 128, 192, 256], [0, 30,  80, 120, 192])

	def colorFilterWarm(self, imgRGB):
		# warming filter: increase red, decrease blue
		c_r,c_g,c_b = cv3.split(imgRGB)
		c_r = cv3.LUT(c_r, self.incrChLUT).astype(np.uint8)
		c_b = cv3.LUT(c_b, self.decrChLUT).astype(np.uint8)
		imgRGB = cv3.merge((c_r,c_g,c_b))

		# increase color saturation
		c_h,c_s,c_v = cv3.split(cv3.cvtColor(imgRGB, cv3.COLOR_RGB2HSV))
		c_s = cv3.LUT(c_s, self.incrChLUT).astype(np.uint8)
		return cv3.cvtColor(cv3.merge((c_h,c_s,c_v)), cv3.COLOR_HSV2RGB)

	def colorFilterCool(self, imgRGB):
		# cooling filter: increase blue, decrease red
		c_r,c_g,c_b = cv3.split(imgRGB)
		c_r = cv3.LUT(c_r, self.decrChLUT).astype(np.uint8)
		c_b = cv3.LUT(c_b, self.incrChLUT).astype(np.uint8)
		imgRGB = cv3.merge((c_r,c_g,c_b))

		# decrease color saturation
		c_h,c_s,c_v = cv3.split(cv3.cvtColor(imgRGB, cv3.COLOR_RGB2HSV))
		c_s = cv3.LUT(c_s, self.decrChLUT).astype(np.uint8)
		return cv3.cvtColor(cv3.merge((c_h,c_s,c_v)), cv3.COLOR_HSV2RGB)

	def cartoonizerInit(self):
		pass

	def cartoonizer(self, imgRGB):
		numDownSamples = 2		# number of downscaling steps
		numBilateralFilters = 7 # number of bilateral filtering steps

		# -- STEP 1 --
		# downsample image using Gaussian pyramid
		imgColor = imgRGB
		for i in xrange(numDownSamples):
			imgColor = cv3.pyrDown(imgColor)
			
		# repeatedly apply small bilateral filter instead of applying
		# one large filter
		for i in xrange(numBilateralFilters):
			imgColor = cv3.bilateralFilter(imgColor, 9, 9, 7)
			
		# upsample image to original size
		for i in xrange(numDownSamples):
			imgColor = cv3.pyrUp(imgColor)

		# -- STEPS 2 and 3 --
		# convert to grayscale and apply median blur
		imgGray = cv3.cvtColor(imgRGB, cv3.COLOR_RGB2GRAY)
		imgBlur = cv3.medianBlur(imgGray, 7)

		# -- STEP 4 --
		# detect and enhance edges
		imgEdge = cv3.adaptiveThreshold(imgBlur, 255, cv3.ADAPTIVE_THRESH_MEAN_C, cv3.THRESH_BINARY, 9, 2)

		# -- STEP 5 --
		# convert back to color so that it can be bit-ANDed with color image
		imgEdge = cv3.cvtColor(imgEdge, cv3.COLOR_GRAY2RGB)
		return cv3.bitwise_and(imgColor, imgEdge)



def main():
	capture = cv3.VideoCapture(0)
	if not(capture.isOpened()):
		capture.open()

	capture.set(cv3.CAP_PROP_FRAME_WIDTH, 640)
	capture.set(cv3.CAP_PROP_FRAME_HEIGHT, 480)

	app = wx.App()
	frame = MyFrame(None, -1, 'chapter1.py', capture)
	frame.Show(True)
	app.MainLoop()

 	# When everything done, release the capture
	capture.release()
	cv3.destroyAllWindows()


if __name__ == '__main__':
	main()
