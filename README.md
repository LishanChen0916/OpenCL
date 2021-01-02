# OpenCL

A histogram is a statistic that shows frequency of a certain occurrence within a data set. The histogram of an image provides a frequency distribution of pixel values in the image. If the image is a color image, the pixel value can be the luminosity value of each pixel or the individual red (R), green (G), and blue (B). More about image histogram can be found at https://en.wikipedia.org/wiki/Image_histogram 
In this problem, we use OpenCL to parallelize the implementation of the image histogram. The input/output is a bitmap image format, which stores R/G/B values of each pixel. See the reference code for how to read/write a bitmap file.

##### Compilation
    g++ histogram.cpp -o histogram -lOpenCL