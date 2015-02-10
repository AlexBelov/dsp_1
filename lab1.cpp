#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int compareInts(const void* a, const void* b)
{
  int arg1 = *reinterpret_cast<const int*>(a);
  int arg2 = *reinterpret_cast<const int*>(b);
  if(arg1 < arg2) return -1;
  if(arg1 > arg2) return 1;
  return 0;
}

Mat medianFilter(Mat image, int window_width, int window_height)
{
  Mat image_new = image.clone();
  int edgex = floor(window_width / 2);
  int edgey = floor(window_height / 2);
  int width = image.rows;
  int height = image.cols;
  int arrlen = window_width * window_height;

  for(int x = edgex; x < width - edgex; x++)
  {
    for(int y = edgey; y < height - edgey; y++)
    {
      int arr[arrlen];
      int i = 0;
      for(int fx = 0; fx < window_width; fx++)
      {
        for(int fy = 0; fy < window_height; fy++)
        {
          Scalar pixel = image.at<uchar>(x + fx - edgex, y + fy - edgey);
          arr[i++] = pixel.val[0];
        }
      }
      qsort(arr, i, sizeof(int), compareInts);
      image_new.at<uchar>(x,y) = arr[arrlen/2];
    }
  }

  return image_new;
}

int otsuThreshold(int *image, int size)
{
  int min=image[0], max=image[0];
  int i, temp, temp1;
  int *hist;
  int histSize;

  int alpha, beta, threshold=0;
  double sigma, maxSigma=-1;
  double w1,a;

  /**** Построение гистограммы ****/
  /* Узнаем наибольший и наименьший полутон */
  for(i=1;i<size;i++)
  {
    temp=image[i];
    if(temp<min)   min = temp;
    if(temp>max)   max = temp;
  }

  histSize=max-min+1;
  if((hist=(int*) malloc(sizeof(int)*histSize))==NULL) return -1;

  for(i=0;i<histSize;i++)
    hist[i]=0;

  /* Считаем сколько каких полутонов */
  for(i=0;i<size;i++)
    hist[image[i]-min]++;

  /**** Гистограмма построена ****/

  temp=temp1=0;
  alpha=beta=0;
  /* Для расчета математического ожидания первого класса */
  for(i=0;i<=(max-min);i++)
  {
    temp += i*hist[i];
    temp1 += hist[i];
  }

  /* Основной цикл поиска порога
  Пробегаемся по всем полутонам для поиска такого, при котором внутриклассовая дисперсия минимальна */
  for(i=0;i<(max-min);i++)
  {
    alpha+= i*hist[i];
    beta+=hist[i];

    w1 = (double)beta / temp1;
    a = (double)alpha / beta - (double)(temp - alpha) / (temp1 - beta);
    sigma=w1*(1-w1)*a*a;

    if(sigma>maxSigma)
    {
      maxSigma=sigma;
      threshold=i;
    }
  }
  free(hist);
  return threshold + min;
}

int getTreshold(Mat image, int x1, int x2, int y1, int y2)
{
  int width = image.rows;
  int height = image.cols;
  int image_otsu[(x2 - x1)*(y2 - y1)];
  int i = 0;

  for(int x = x1; x < x2; x++)
  {
    for(int y = y1; y < y2; y++)
    {
      Scalar pixel = image.at<uchar>(x, y);
      image_otsu[i++] = pixel.val[0];
    }
  }
  int threshold = otsuThreshold(image_otsu, i);
  return threshold;
}

Mat adaptiveBinarization(Mat image)
{
  Mat image_binarized = image.clone();
  int threshold = 0;
  int width = image.rows;
  int height = image.cols;
  int window_width = 10;
  int window_height = 10;

  for(int x = 0; x < width; x = x + window_width)
  {
    for(int y = 0; y < height; y = y + window_height)
    {
      //int x1 = x - window_width/2;
      int x2 = x + window_width;
      //int y1 = y - window_height/2;
      int y2 = y + window_height;

      // if(x1 < 0)
      //   x1 = 0;
      if(x2 > width)
        x2 = width;
      // if(y1 < 0)
      //   y1 = 0;
      if(y2 > height)
        y2 = height;

      threshold = getTreshold(image, x, x2, y, y2);
      threshold = 210;

      for(int fx = x; fx < x2; fx++)
      {
        for(int fy = y; fy < y2; fy++)
        {
          Scalar pixel = image.at<uchar>(fx, fy);
          if(pixel.val[0] >= threshold)
          {
            image_binarized.at<uchar>(fx, fy) = 255;
          }
          else
          {
            image_binarized.at<uchar>(fx, fy) = 0;
          }
        }
      }
    }
  }

  return image_binarized;
}

Mat Binarization(Mat image, int threshold)
{
  Mat image_binarized = image.clone();
  int width = image.rows;
  int height = image.cols;

  for(int x = 0; x < width; x++)
  {
    for(int y = 0; y < height; y++)
    {
      Scalar pixel = image.at<uchar>(x, y);
      if(pixel.val[0] >= threshold)
      {
        image_binarized.at<uchar>(x, y) = 255;
      }
      else
      {
        image_binarized.at<uchar>(x, y) = 0;
      }
    }
  }
  return image_binarized;
}

Mat morphologyErosion(Mat image, int* mask, int mask_size)
{
  int width = image.rows;
  int height = image.cols;
  Mat image_erosion = image.clone();

  for(int y = mask_size/2; y < height; y++)
  {
    for(int x = mask_size/2; x < width; x++)
    {
      Scalar pixel = image.at<uchar>(x, y);
      if(pixel.val[0] == 255)
      {
        int zero_present = 0;
        for(int fx = x-mask_size/2; fx <= x+mask_size/2; fx++)
        {
          for(int fy = y-mask_size/2; fy <= y+mask_size/2; fy++)
          {
            Scalar pixel2 = image.at<uchar>(fx, fy);
            if(pixel2.val[0] == 0)
            {
              zero_present = 1;
            }
          }
        }
        if(zero_present == 1)
        {
          int i = 0;
          for(int fy = y-mask_size/2; fy <= y+mask_size/2; fy++)
          {
            for(int fx = x-mask_size/2; fx <= x+mask_size/2; fx++)
            {
              int mask_element = mask[i++];
              if(mask_element == 1)
              {
                image_erosion.at<uchar>(fx, fy) = 0;
              }
            }
          }
        }
      }
    }
  }

  return image_erosion;
}

Mat morphologyDilation(Mat image, int* mask, int mask_size)
{
  int width = image.rows;
  int height = image.cols;
  Mat image_dilation = image.clone();

  for(int y = mask_size/2; y < height; y++)
  {
    for(int x = mask_size/2; x < width; x++)
    {
      Scalar pixel = image.at<uchar>(x, y);
      if(pixel.val[0] == 255)
      {
        int zero_present = 0;
        for(int fx = x-mask_size/2; fx <= x+mask_size/2; fx++)
        {
          for(int fy = y-mask_size/2; fy <= y+mask_size/2; fy++)
          {
            Scalar pixel2 = image.at<uchar>(fx, fy);
            if(pixel2.val[0] == 0)
            {
              zero_present = 1;
            }
          }
        }
        if(zero_present == 1)
        {
          int i = 0;
          for(int fy = y-mask_size/2; fy <= y+mask_size/2; fy++)
          {
            for(int fx = x-mask_size/2; fx <= x+mask_size/2; fx++)
            {
              int mask_element = mask[i++];
              if(mask_element == 1)
              {
                image_dilation.at<uchar>(fx, fy) = 255;
              }
            }
          }
        }
      }
    }
  }

  return image_dilation;
}

int main(int argc, char** argv)
{
  if(argc != 2)
  {
    cout <<" Usage: ./lab1 image_number" << endl;
    return -1;
  }

  Mat image;

  string image_path;
  string new_image_path;
  image_path = string("img/") + argv[1] + ".jpg";
  new_image_path = string("img/") + argv[1] + "_new.jpg";

  image = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
  int width = image.rows;
  int height = image.cols;

  image = medianFilter(image, 3, 3);
  //image = adaptiveBinarization(image);
  int thresholds[5] = {215,215,210,200,190};
  image = Binarization(image, thresholds[atoi(argv[1]) - 1]);

  int dilation_mask[25] = {
    0,1,1,1,0,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1,
    0,1,1,1,0
  };

  int erosion_mask[9] = {
    1,1,1,
    0,1,0,
    0,1,0
  };

  image = morphologyErosion(image, erosion_mask, 3);
  image = medianFilter(image, 3, 3);
  image = morphologyDilation(image, dilation_mask, 5);
  //image = morphologyDilation(image, dilation_mask, 5);
  //image = morphologyDilation(image, dilation_mask, 5);

  imwrite(new_image_path, image);
  namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
  imshow("Display window", image);                   // Show our image inside it.

  waitKey(0);                                          // Wait for a keystroke in the window
  return 0;
}
