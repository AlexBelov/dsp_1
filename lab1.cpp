#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <cmath>
#include <set>

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

int compareDoubles(const void* a, const void* b)
{
  int arg1 = *reinterpret_cast<const double*>(a);
  int arg2 = *reinterpret_cast<const double*>(b);
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
  Mat image_erosion = Mat(width,height, CV_8UC3, Scalar(0));

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
        else
        {
          image_erosion.at<uchar>(x, y) = image.at<uchar>(x, y);
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
  Mat image_dilation = Mat(width,height, CV_8UC3, Scalar(0));

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
        else
        {
          image_dilation.at<uchar>(x, y) = image.at<uchar>(x, y);
        }
      }
    }
  }

  return image_dilation;
}

void fillArea(Mat image, int* labels, int x, int y, int contour)
{
  int width = image.rows;
  int height = image.cols;
  Scalar pixel = image.at<uchar>(x, y);

  if(labels[y*width + x] == 0 && pixel.val[0] == 255)
  {
    labels[y*width + x] = contour;
    if(x > 0)
      fillArea(image, labels, x-1, y, contour);
    if(x < width-1)
      fillArea(image, labels, x+1, y, contour);
    if(y > 0)
      fillArea(image, labels, x, y-1, contour);
    if(y < height-1)
      fillArea(image, labels, x, y+1, contour);
  }
}

void recursiveLabeling(Mat image, int* labels)
{
  int contour = 1;
  int width = image.rows;
  int height = image.cols;

  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      fillArea(image, labels, x, y, contour++);
    }
  }
}

Mat colorizeContours(Mat image, int* labels)
{
  int width = image.rows;
  int height = image.cols;
  Mat image_colorized = Mat(width,height, CV_8UC3, Scalar(0,0,0));
  int i = 0;

  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      if(labels[i] != 0)
      {
        Vec3b color = image_colorized.at<Vec3b>(Point(y,x));
        color[0] = 50*labels[i];
        color[1] = 30*labels[i];
        color[2] = 40*labels[i];
        image_colorized.at<Vec3b>(Point(y,x)) = color;
      }
      i++;
    }
  }

  return image_colorized;
}

void normalizeLabels(int* labels, int* contours, int labels_size, int contours_size, char* pic)
{
  for(int i = 0; i < labels_size; i++)
  {
    int value = labels[i];
    if(value != 0)
    {
      int index = std::distance(contours, std::find(contours, contours + contours_size, value));
      labels[i] = index + 1;
      if(*pic == '3' || *pic == '4') { labels[i]--; }
    }
  }
}

void getS(int* labels, int* S, int labels_size, int image_size)
{
  for(int i = 0; i < image_size; i++)
  {
    if(labels[i] != 0)
    {
      S[labels[i]] += 1;
    }
  }
}

void getP(int* labels, int* P, int labels_size, int width, int height)
{
  int value = 0;
  int flag = 0;

  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      value = labels[y*width + x];
      flag = 0;
      if(value != 0)
      {
        for(int fy = y-1; fy < y+1; fy++)
        {
          for(int fx = x-1; fx < x+1; fx++)
          {
            if(labels[fy*width + fx] != value)
            {
              flag = 1;
            }
          }
        }
        if(flag == 1)
        {
          P[value] += 1;
        }
      }
    }
  }
}

void getC(int* P, int* S, double* C, int size)
{
  for(int i = 1; i < size + 1; i++)
  {
    C[i] = double(P[i]*P[i]/double(S[i]));
  }
}

void getCenterOfMass(int* labels, int width, int height, int contour_index, int& x_mass_center, int& y_mass_center)
{
  int value = 0;
  int sum_x = 0;
  int sum_y = 0;
  int S = 0;

  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      value = labels[y*width + x];
      if(value == contour_index)
      {
        sum_x += x;
        sum_y += y;
        S += 1;
      }
    }
  }

  x_mass_center = int(sum_x/S);
  y_mass_center = int(sum_y/S);
}

double getDiscreteCentralMoment(int* labels, int width, int height, int contour_index, int i, int j)
{
  int x_mass_center = 0;
  int y_mass_center = 0;
  int value = 0;
  double result = 0;

  getCenterOfMass(labels, width, height, contour_index, x_mass_center, y_mass_center);
  //cout << x_mass_center << " " << y_mass_center << endl;

  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      value = labels[y*width + x];
      result += pow(x - x_mass_center, i)*pow(y - y_mass_center, j);
    }
  }

  return result;
}

double getElongation(int* labels, int width, int height, int contour_index)
{
  double m20 = getDiscreteCentralMoment(labels, width, height, contour_index, 2, 0);
  double m02 = getDiscreteCentralMoment(labels, width, height, contour_index, 0, 2);
  double m11 = getDiscreteCentralMoment(labels, width, height, contour_index, 1, 1);

  double result = m20+m02+sqrt(pow(m20-m02,2)+4*pow(m11,2));
  result /= m20+m02-sqrt(pow(m20-m02,2)+4*pow(m11,2));

  return result;
}

void getE(int* labels, double* E, int labels_size, int width, int height)
{
  for(int i = 1; i < labels_size + 1; i++)
  {
    E[i] = getElongation(labels, width, height, i);
  }
}

double getDistanceToCenter(double* feature, double* cluster_center, int num_features)
{
  double result = 0;
  for(int i = 0; i < num_features; i++)
  {
    result += pow(feature[i] - cluster_center[i], 2);
  }
  result = sqrt(result);
  //cout << "Distance: " << result << endl;
  return result;
}

int getFeaturesClusters(double** features, double** cluster_centers, int* feature_clusters, int contour_num, int features_num, int num_clusters)
{
  double min_distance = 0;
  int nearest_cluster = 0;
  double distance = 0;
  int changed = 0;

  for(int i = 1; i < contour_num + 1; i++)
  {
    min_distance = getDistanceToCenter(features[i], cluster_centers[0], features_num);
    nearest_cluster = 0;

    for(int j = 0; j < num_clusters; j++)
    {
      distance = getDistanceToCenter(features[i], cluster_centers[j], features_num);
      if(distance < min_distance)
      {
        min_distance = distance;
        nearest_cluster = j;
      }
    }

    if(feature_clusters[i] != nearest_cluster)
    {
      changed = 1;
    }
    feature_clusters[i] = nearest_cluster;
  }

  return changed;
}

void alignClusterCenters(double** features, double** cluster_centers, int* feature_clusters, int contour_num, int num_features, int num_clusters)
{
  int nearest_cluster = 0;
  double* new_center = (double*)calloc(num_features, sizeof(double));
  int num_contours_in_cluster = 0;

  for(int i = 0; i < num_clusters; i++)
  {
    for(int k = 0; k < num_features; k++)
    {
      new_center[k] = 0;
    }

    num_contours_in_cluster = 0;

    for(int j = 1; j < contour_num + 1; j++)
    {
      nearest_cluster = feature_clusters[j];
      if (nearest_cluster == i)
      {
        for(int k = 0; k < num_features; k++)
        {
          new_center[k] += features[j][k];
        }

        num_contours_in_cluster++;
      }
    }

    for(int k = 0; k < num_features; k++)
    {
      new_center[k] /= num_contours_in_cluster;
      cluster_centers[i][k] = new_center[k];
    }
  }
}

void kMeans(double** features, int* feature_clusters, int num_clusters, int features_num, int contour_num)
{
  double** cluster_centers = (double**)calloc(num_clusters, sizeof(double*));
  for(int i = 0; i < num_clusters; i++)
  {
    cluster_centers[i] = (double*)calloc(features_num, sizeof(double));
  }

  cluster_centers[0][0] = 1000;
  cluster_centers[0][1] = 70;
  cluster_centers[0][2] = 6;
  cluster_centers[0][3] = 2.7;

  cluster_centers[1][0] = 4300;
  cluster_centers[1][1] = 300;
  cluster_centers[1][2] = 20;
  cluster_centers[1][3] = 2.1;

  if(num_clusters - 2 > 0)
  {
    for(int i = 2; i < num_clusters; i++)
    {
      cluster_centers[i][0] = rand() % 6000 + 100;
      cluster_centers[i][1] = rand() % 400 + 50;
      cluster_centers[i][2] = rand() % 30 + 3;
      cluster_centers[i][3] = rand() % 5;
    }
  }

  // cout << "Cluster centers:" << endl;
  // for(int i = 0; i < num_clusters; i++)
  // {
  //   cout << cluster_centers[i][0] << " ";
  //   cout << cluster_centers[i][1] << " ";
  //   cout << cluster_centers[i][2] << " ";
  //   cout << cluster_centers[i][3] << " " << endl;
  // }

  int changed = 1;

  while(changed == 1)
  {
    changed = getFeaturesClusters(features, cluster_centers, feature_clusters, contour_num, features_num, num_clusters);
    if(changed == 0)
    {
      break;
    }
    alignClusterCenters(features, cluster_centers, feature_clusters, contour_num, features_num, num_clusters);
  }

  // for(int i = 0; i < num_clusters; i++)
  // {
  //   free(cluster_centers[i]);
  // }
  // free(cluster_centers);
}

Mat colorizeClusters(Mat image, int* labels, int* clusters, int** colors)
{
  int width = image.rows;
  int height = image.cols;
  Mat image_colorized = Mat(width,height, CV_8UC3, Scalar(0,0,0));
  int i = 0;

  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      if(labels[i] != 0)
      {
        Vec3b color = image_colorized.at<Vec3b>(Point(y,x));
        color[0] = colors[clusters[labels[i]]][0];
        color[1] = colors[clusters[labels[i]]][1];
        color[2] = colors[clusters[labels[i]]][2];
        //cout << labels[i] << " " << clusters[labels[i]-1] << endl;
        image_colorized.at<Vec3b>(Point(y,x)) = color;
      }
      i++;
    }
  }

  return image_colorized;
}

bool inArray(int* array, int size, int num)
{
  for(int i = 0; i < size; i++)
  {
    if(array[i] == num)
    {
      return true;
    }
  }
  return false;
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

  image = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
  int width = image.rows;
  int height = image.cols;

  image = medianFilter(image, 3, 3);
  //image = adaptiveBinarization(image);
  int thresholds[10] = {215,235,220,160,200,200,200,200,200,212};
  image = Binarization(image, thresholds[atoi(argv[1]) - 1]);

  int dilation_mask[25] = {
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1
  };

  int erosion_mask_1[9] = {
    1,1,1,
    1,1,1,
    1,1,1
  };

  int erosion_mask_2[25] = {
    0,0,0,0,0,
    0,0,1,0,0,
    0,0,1,0,0,
    0,0,1,0,0,
    0,0,0,0,0
  };

  if(*argv[1] == '2' || *argv[1] == '4')
  {
    image = morphologyErosion(image, erosion_mask_2, 5);
  }
  else
  {
    image = morphologyErosion(image, erosion_mask_1, 3);
  }

  image = medianFilter(image, 3, 3);
  image = morphologyDilation(image, dilation_mask, 5);

  int* labels = (int*)calloc(width*height,sizeof(int));
  recursiveLabeling(image, labels);

  std::set<int> slabels(labels, labels + width*height);

  int contour_num = 0;

  if(*argv[1] == '3' || *argv[1] == '4')
  {
    contour_num = slabels.size() - 2;
  }
  else
  {
    contour_num = slabels.size() - 1;
  }

  cout << "Contours number: " << contour_num << endl;

  int contours[contour_num];
  int i = 0;
  for(set<int>::iterator elementIt = slabels.begin(); elementIt != slabels.end(); elementIt++)
  {
    int x = *elementIt;
    if(x != 0)
      contours[i++] = x;
  }

  normalizeLabels(labels, contours, width*height, contour_num, argv[1]);
  // image = colorizeContours(image, labels);
  // imwrite("img/col.jpg", image);

  //image = colorizeContours(image, labels);

  int* S = (int*)calloc(contour_num + 1,sizeof(int));
  getS(labels, S, contour_num, width*height);

  int* P = (int*)calloc(contour_num + 1,sizeof(int));
  getP(labels, P, contour_num, width, height);

  double* C = (double*)calloc(contour_num + 1,sizeof(double));
  getC(P, S, C, contour_num);

  double* E = (double*)calloc(contour_num + 1,sizeof(double));
  getE(labels, E, contour_num, width, height);

  double** features = (double**)calloc(contour_num + 1, sizeof(double*));
  int features_num = 4;

  for(int i = 0; i < contour_num + 1; i++)
  {
    features[i] = (double*)calloc(features_num, sizeof(double));
    features[i][0] = S[i];
    features[i][1] = P[i];
    features[i][2] = C[i];
    features[i][3] = E[i];
  }

  int num_clusters = 2;
  int* feature_clusters = (int*)calloc(contour_num + 1, sizeof(int));

  kMeans(features, feature_clusters, num_clusters, features_num, contour_num);

  // for(int i = 1; i < contour_num + 1; i++)
  // {
  //   cout << feature_clusters[i] << " ";
  // }
  // cout << endl;

  cout << "Features vector" << endl;
  cout << "  S   P     C       E" << endl;
  for(int i = 1; i < contour_num + 1; i++)
  {
    cout << setw(4);
    cout << features[i][0] << " ";
    cout << setw(3);
    cout << features[i][1] << " ";
    cout << setw(7);
    cout << features[i][2] << " ";
    cout << setw(7);
    cout << features[i][3] << " ";
    cout << setw(7);
    cout << "Cluster: " << feature_clusters[i];
    cout << endl;
  }
  cout << endl;

  int** colors = (int**)calloc(num_clusters + 1, sizeof(int*));
  for(int i = 0; i < num_clusters + 1; i++)
  {
    colors[i] = (int*)calloc(3, sizeof(int));
    colors[i][0] = rand() % 255;
    colors[i][1] = rand() % 255;
    colors[i][2] = rand() % 255;
  }

  int* feature_uniq = (int*)calloc(contour_num + 1, sizeof(int));
  int uniq_features_num = 0;

  for(int i = 1; i < contour_num + 1; i++)
  {
    if(!inArray(feature_uniq, contour_num + 1, feature_clusters[i]))
    {
      feature_uniq[uniq_features_num++] = feature_clusters[i];
    }
  }

  double* median_features = (double*)calloc(contour_num + 1, sizeof(double));
  int num_features_median = 0;
  double median_value = 0;
  int feature_for_filter_num = 1;
  int feature_filter = 1;
  double up_threshold = 1.1;

  for(int feature_filter = 0; feature_filter < feature_for_filter_num; feature_filter++)
  {
    for(int i = 0; i < uniq_features_num + 1; i++)
    {
      num_features_median = 0;
      for(int j = 1; j < contour_num + 1; j++)
      {
        if(feature_clusters[j] == feature_uniq[i])
        {
          median_features[num_features_median++] = features[j][feature_filter];
        }
      }
      qsort(median_features, num_features_median, sizeof(double), compareDoubles);
      median_value = median_features[num_features_median/2];

      if (*argv[1] == '3') { up_threshold = 1.26; }
      if (*argv[1] == '7') { up_threshold = 1.7; }

      // for(int i = 1; i < contour_num + 1; i++)
      // {
      //   cout << feature_clusters[i] << " ";
      // }
      // cout << endl;
      //cout << median_value << endl;
      for(int j = 1; j < contour_num + 1; j++)
      {
        if(feature_clusters[j] == feature_uniq[i])
        {
          if(features[j][feature_filter] > median_value*up_threshold || features[j][feature_filter] < median_value*0.7)
          {
            feature_clusters[j] = uniq_features_num + 1;
          }
        }
      }
    }
  }

  cout << "Features vector" << endl;
  cout << "  S   P     C       E" << endl;
  for(int i = 1; i < contour_num + 1; i++)
  {
    cout << setw(4);
    cout << features[i][0] << " ";
    cout << setw(3);
    cout << features[i][1] << " ";
    cout << setw(7);
    cout << features[i][2] << " ";
    cout << setw(7);
    cout << features[i][3] << " ";
    cout << setw(7);
    cout << "Cluster: " << feature_clusters[i];
    cout << endl;
  }
  cout << endl;

  image = colorizeClusters(image, labels, feature_clusters, colors);

  imwrite(new_image_path, image);
  namedWindow("Display window", WINDOW_AUTOSIZE);
  imshow("Display window", image);

  waitKey(0);

  //free(feature_clusters);
  return 0;
}
