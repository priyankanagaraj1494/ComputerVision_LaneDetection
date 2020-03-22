# Lane_detection

### Lane detection using computer vision

* The images are captured from CARLA Simulator using a RGB sensor that is attached to the vehicle </br> 
* These RGB images(1280, 720,3) are converted to grayscale image (1280, 720) </br> 
* Gaussian filter of size (5,5) is applied to grayscale image to remove noise by blurring the image </br>
* Canny edge detector algorithm is apllied to find the edge in the image </br>
* The region with lane is selected by a polygon and rest of the image is set to black </br> 
* The masked image is drawn with hough lines which is later combined with original image </br>

![The ouput images of each step](./images/plot.PNG)
![Image from CARLA simulator](./data/00002498.png)
![Lane detected](./images/lane.PNG)

![Car with rgb sensor](./images/carla_driving.png)   ![image captured from sensor](./data/test.png) 
![Lane detected from captured image](./images/lane_whiledriving.PNG)
