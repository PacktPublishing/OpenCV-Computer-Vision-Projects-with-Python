#!/bin/bash

arch=$(uname -m)
if [ "$arch" == "i686" -o "$arch" == "i386" -o "$arch" == "i486" -o "$arch" == "i586" ]; then
    is_x86=1
else
    is_x86=0
fi

echo "Preparing to install OpenNI 1.5.4.0 (unstable), SensorKinect 0.93, OpenCV 2.4.3 and dependencies"
mkdir opencv
cd opencv

echo "Installing dependencies of OpenNI"
sudo apt-get install libusb-1.0-0-dev freeglut3-dev

echo "Downloading OpenNI"
if [ $is_x86 -eq 1 ]; then
    wget http://www.openni.org/wp-content/uploads/2012/12/OpenNI-Bin-Dev-Linux-x86-v1.5.4.0.tar.zip
    unzip OpenNI-Bin-Dev-Linux-x86-v1.5.4.0.tar.zip
    tar -xvf OpenNI-Bin-Dev-Linux-x86-v1.5.4.0.tar.bz2
    cd OpenNI-Bin-Dev-Linux-x86-v1.5.4.0
else
    wget http://www.openni.org/wp-content/uploads/2012/12/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0.tar.zip
    unzip OpenNI-Bin-Dev-Linux-x64-v1.5.4.0.tar.zip
    tar -xvf OpenNI-Bin-Dev-Linux-x64-v1.5.4.0.tar.bz2
    cd OpenNI-Bin-Dev-Linux-x64-v1.5.4.0
fi
echo "Installing OpenNI"
sudo chmod a+x install.sh
sudo ./install.sh
cd ..

echo "Downloading SensorKinect"
if [ $is_x86 -eq 1 ]; then
    wget --no-check-certificate https://github.com/avin2/SensorKinect/blob/unstable/Bin/SensorKinect093-Bin-Linux-x86-v5.1.2.1.tar.bz2?raw=true
    tar -xvf SensorKinect093-Bin-Linux-x86-v5.1.2.1.tar.bz2?raw=true
    cd Sensor-Bin-Linux-x86-v5.1.2.1
else
    wget --no-check-certificate https://github.com/avin2/SensorKinect/blob/unstable/Bin/SensorKinect093-Bin-Linux-x64-v5.1.2.1.tar.bz2?raw=true
    tar -xvf SensorKinect093-Bin-Linux-x64-v5.1.2.1.tar.bz2?raw=true
    cd Sensor-Bin-Linux-x64-v5.1.2.1
fi
echo "Installing SensorKinect"
sudo chmod a+x install.sh
sudo ./install.sh
cd ..

echo "Removing any pre-installed ffmpeg and x264"
sudo apt-get remove ffmpeg x264 libx264-dev

echo "Installing dependenices of OpenCV"
sudo apt-get install libopencv-dev
sudo apt-get install build-essential checkinstall cmake pkg-config yasm
sudo apt-get install libtiff4-dev libjpeg-dev libjasper-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev
sudo apt-get install python-dev python-numpy python-scipy
sudo apt-get install libtbb-dev
sudo apt-get install libqt4-dev libgtk2.0-dev

echo "Downloading x264"
wget ftp://ftp.videolan.org/pub/videolan/x264/snapshots/x264-snapshot-20120528-2245-stable.tar.bz2
tar -xvf x264-snapshot-20120528-2245-stable.tar.bz2
cd x264-snapshot-20120528-2245-stable/
echo "Installing x264"
if [ $is_x86 -eq 1 ]; then
    ./configure --enable-static
else
    ./configure --enable-shared --enable-pic
fi
make
sudo make install
cd ..

echo "Downloading ffmpeg"
wget http://ffmpeg.org/releases/ffmpeg-0.11.1.tar.bz2
echo "Installing ffmpeg"
tar -xvf ffmpeg-0.11.1.tar.bz2
cd ffmpeg-0.11.1/
if [ $is_x86 -eq 1 ]; then
    ./configure --enable-gpl --enable-libfaac --enable-libmp3lame --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libtheora --enable-libvorbis --enable-libx264 --enable-libxvid --enable-nonfree --enable-postproc --enable-version3 --enable-x11grab
else
    ./configure --enable-gpl --enable-libfaac --enable-libmp3lame --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libtheora --enable-libvorbis --enable-libx264 --enable-libxvid --enable-nonfree --enable-postproc --enable-version3 --enable-x11grab --enable-shared
fi
make
sudo make install
cd ..

echo "Downloading v4l"
wget http://www.linuxtv.org/downloads/v4l-utils/v4l-utils-0.8.8.tar.bz2
echo "Installing v4l"
tar -xvf v4l-utils-0.8.8.tar.bz2
cd v4l-utils-0.8.8/
make
sudo make install
cd ..

echo "Downloading OpenCV"
wget -O OpenCV-2.4.3.tar.bz2 http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2/download
echo "Installing OpenCV"
tar -xvf OpenCV-2.4.3.tar.bz2
cd OpenCV-2.4.3
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENNI=ON ..
make
sudo make install
echo "OpenCV is installed to /usr/local/lib"
echo "Appending \"/usr/local/lib\" to /etc/ld.so.conf"
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf
sudo ldconfig
echo "OpenCV ready to be used"