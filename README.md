# Lane_detection

### Lane detection using computer vision

* The images are captured from CARLA Simulator using a RGB sensor that is attached to the vehicle </br> 
* These RGB images(1280, 720,3) are converted to grayscale image (1280, 720) </br> 
* Gaussian filter of size (5,5) is applied to grayscale image to remove noise by blurring the image </br>
* Canny edge detector algorithm is apllied to find the edge in the image </br>
* The region with lane is selected by a polygon and rest of the image is set to black </br> 
* The masked image is drawn with hough lines which is later combined with original image </br>

![The ouput images of each step](./images/plot.PNG)

#### The following image shows the lane detected using lane_detection_canny.py.

<p float="left">
  <img src="./data/00002498.png" alt="image_"	title="image from simulator" width="300" height="200" />
  <img src="./_out/lane.PNG" alt="lane_"	title="Lane detected" width="300" height="200" /> 
</p>

### Simultaneously reading and detecting lanes

The manual_control.py is a program to control the car in CARLA simulator. Refer http://carla.org/ for more details. The first image is the car with a RGB camera sensor is attached to record images. The second image is the image captured from the sensor. This image is passed to carla_lane.py function to detect lanes. The third image is the image with lane markings are saved in folder *_out*. 

<p float="left">
  <img src="./images/carla_driving.png" alt="car"	title="Car with RGB sensor" width="250" height="200" />
  <img src="./data/test.png" alt="image"	title="Image captured from sensor" width="250" height="200" /> 
  <img src="./_out/lane_whiledriving.PNG" alt="lane"	title="Lane detected" width="250" height="200" />
</p>
