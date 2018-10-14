## Application and Evaluation of Sparse to Dense Depth Estimating Method
============================

### Application Setup  

We developed the application based on the original code from [1]. It can achieve
depth constructing from live image input or pre-recorded measurements, with the
output consisting of five images stacked horizontally, which includes RGB images,
depth prediction, raw depth input, sparse depth input and error heat-map. There
are two stand-alone python files named CameraCapture.py and DepthPrediction.py,
with both files needing to be located at the root directory of the [1] project file.
CameraCapture.py is used to record raw RGB and depth data from Kinect and saved
to file ./CameraData, whereas the prediction is implemented in DepthPrediction.py.
Details of the code structures are discussed in Section 1.2 and 1.3. The details of the
usage of the program is listed in Section 1.4.

#### 1.1 Prerequisite
The implementation is dedicated on Windows platform, mainly due to the reason
of Kinect support which is required for accessing live RGB and depth frames.
Hardware requirements include: a Windows machine with CUDA-enabled GPU (at
least 4GB of RAM) and Mircosoft Kinect for Xbox One with USB adaptor. Since
the code was developed and tested using Python3.6, we strongly recommend users
to install Python3. Several libraries including cv2, pickle, torch, numpy, mat-
plotlib and pykinect2 are required. Note on the pykinect2 that the pip version
does not support 64-bit system, thus if pykinect2 is installed through pip, one
should replace the PyKinectV2 file from the system python library (e.g. in windows
C:n...nPythonnPython36nLibnsite-packagesnpykinect2nPyKinectV2.py) with the updated
file from github: https://github.com/Kinect/PyKinect2/blob/master/
pykinect2/PyKinectV2.py. This implementation is based on the project [18], so
the original code should be downloaded from https://github.com/fangchangma/
sparse-to-dense.pytorch. One should also download and install the ’Kinect for
Windows SDK v2’ as the Kinect driver.  

#### 1.2 CameraCapture.py

Although the aim of this project is to build a live depth prediction tool, in order to
save and review the input and prediction, it is better to store a single frame from
the video stream and then perform a depth estimation on that single frame. The
purpose of this code is to save the single frame from the Kinect input stream. The
code uses the PyKinect library to access both colour and depth frame, and once both
frames are available, they will be displayed to allow viewing of the current scene.
Notice that the original Kinect RGB data is stored in BGRA format, thus additional
conversion from BGRA to BGR is applied using cv2 function:
cvtColor(RGBA; cv2:COLOR RGBA2RGB)
The depth measurement is in mm, to simply convert it to greyscale (values between
0-255) for visualising, it is clipped in the range between (0, 5100) and then divided
by 20. An example of such images is shown below.
The program will continuously display the latest frame fetched from the Kinect.
Meanwhile it is also detecting for key ’ESC’, and once the key is pressed, the program
will save the current depth and RGB frame to a .pickle file under ./CameraData and
then exit. The file name of the pickle file is the same as the second argument of the
run command, thus a new frame with the same name as any previous frame will
simply overwrite the earlier file.  

#### 1.3 DepthPrediction.py

The main depth predictions are implemented in this file. The main function will be
to initially load the pre-trained model in the :=result file based on the command line
arguments. Currently, only nyudeothv2 is accepted as a dataset argument. After
that, the program will split into three branches according to the input arguments
’–camera’, it specifies the input source, thus turning the program into three modes:
kinect, webcam and recorded data.

**Kinect Mode**
The Kinect Mode means that the program will fetch the input from the Kinect and
display live depth predictions. The program will initially detect if there is a Kinect
device connected. After that the program will save both RGB (converted from RGBA)
and depth frame, and pass these to the DepthEstimate() function, which will be
explained in the next section. DepthEstimate() function will take two arguments,
RGB data and depth data, and returns with a image and corresponding RMSE value
for that frame. The image is then displayed. One in every 100 frames, the RMSE
value will be printed.
**Record Data Mode**
For the record data mode, instead of using live frames from the Kinect device, it
opens pickle files saved by CameraCapture:py and extracts the RGB and depth data.
Similar to Kinect Mode, both modes call the DepthEstimate() function to estimate
depth prediction; however, in this mode, the DepthEstimate() function is only a onetime-
operation since we are dealing with a single frame. Also, the 0save0 variable is
set to True to save output.
**Webcam Mode**
Unlike the other two modes, the Webcam mode uses RGB data from the webcam
as input, the overall process being very similar to the codes implemented in the
DepthEstimate() function. However, only RGB images and depth predictions are
displayed because we only have these sources available.

#### 1.4 DepthEstimate()

The DepthEstimate() takes five arguments: trained model, RGB frame, depth frame
and save and a bool variable. The variable save determines if the images will be
saved, while the last bool variable actives printing of the number of depth samples .
Firstly, since the RGB frame and depth frame are different sizes, 1920x1080 for
the colour image and 512x424 for the depth frame, both input frames need to be
downsampled to the size 304228, which is the size of the input layer of the CNN
model. The two cameras in Kinect have a different Field of View (FOV), as can
be seen in Figure 4.2, which makes it difficult to align two frames. Therefore, we
decided to first downsample both frames to the same height of 240, so the RGB
frame is now 426x240 and the depth frame is 320x240, with centre-cropping then
performed on both frames to ensure coverage of both cameras.  
Different from training where the sparse depth was randomly sampled from the raw
depth input, in a real-life scenario, sparse depth samples are usually fixed pixels (in
the case of LiDAR, all depth pixels are fixed). If we use randomly sampled depth
as input, the depth prediction will fluctuate heavily due to the reason of random
depth pixel position and the heavy influence of the depth measurement. Therefore,
we decided to evenly sample the depth measurement to create our sparse depth
sample.  
In case of RGBD modality, the sparse depth samples will be added to the RGB images
as 4th dimensions. Otherwise, it will either be used as input in depth-only modality
or simply be discarded in RGB modality.  
The input array will then be converted to torch’s tensor form and fed to the model.
After computing the depth prediction, all depth data will be initially normalised and
then converted to a colour map using matplotlib.pyplot.cm function.
The error heat-map is one way to visualise the difference between the prediction and
groundtruth. It is generated by first covering the missing data from the raw depth
measurement using depth prediction, then computing the absolute difference between
raw depth and depth prediction. After normalising the absolute depth difference
map, we use the same matplotlib.pyplot.cm function to generate the error-heat
map.  
We stack RGB frame, depth prediction colour map, raw depth colour map, sparse
depth map and error heat-map together to form the final result, as well as the return
value of the DepthEstimate() function.  
If images are required to be saved, it will automatically save all colour map images
produced to the corresponding directory in the output folder.

#### 1.5 Usage

Firstly, ensure that all hardware requirements in the prerequisites are met, with all
the software and dependencies successfully downloaded and installed. Since the
trained model is too large and exceeds the file submitting limit, we do not include
it in the archive.zip file. There are two ways to acquire the trained model:1. Train
the model locally using the [1] program. 2. Download from google cloud using the
link in Appendix I. If using method 1 please refer to github page https://github.com/fangchangma/sparse-to-dense.pytorch
, otherwise unzip the downloaded file
and copy the file named   

**nyudepthv2.sparsifier=uar.samples=500.modality=rgbd.arch=resnet50.decoder=upproj.criterion=l1.lr=0.01.bs=8.pretrained=True**

to the [1] project program under the ./result folder.
The test scripts come with several options, which can be listed with the –help flag.

**py DepthPrediction.py –help**

There are 8 essential arguments required to successfully run the script, with an example
being:

**py DepthPrediction.py -a resnet50 -d upproj -m rgbd -s 500 –data nyudepthv2 –camera record -ss 18 –kinectdata lab raw 1**

which means using the model trained with RGBD information and 500 sparse depth
samples on nyudepthv2, performing prediction on pre-recorded data ’lab raw 1’ and
the sparse sampling space is 18 pixels(taking 1 sparse depth sample in 18 pixels).
By adding the ’-w’ command the output image will be saved as the .nimage result.
To generate Kinect raw measurement, simply run:  

**py CameraCapture.py NAME**  

NAME will be the file name for the current scene measurement.



[1] F. Ma and S. Karaman. Sparse-to-dense: Depth prediction from sparse depth
samples and a single image. CoRR, abs/1709.07492, 2017.

