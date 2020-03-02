#include <iostream>
#include "bitmap_image.hpp"
#include <math.h>

const int MAX_ITERATION = 255;

__global__
void cudaCalc(int w, int h, double mx, double my, double zoom, float *results){
  int index = threadIdx.x;
  int stride = blockDim.x;

  for(int px = index; px < w; px += stride){
    for(int py = 0; py < h; py++){
      
      double scaledX = ((double)px - w / 2) / (w / 2) / zoom + mx;
      double scaledY = ((double)py - h / 2) / (h / 2) / zoom + my;
    
      double x = 0;
      double y = 0;
  
      float it = 0;
  
      while (x * x + y * y < 4 && it < MAX_ITERATION)
      {
          double xTemp = x * x - y * y + scaledX;
          y = 2 * x * y + scaledY;
          x = xTemp;
  
          it++;
      }

      results[px * w + py] = it;
    }
  }
}

int main(void){
    int w = 1000;
    int h = 1000;

    float xpos = -1.1;
    float ypos = 0;

    double z = (int)1;

    while(true)
    {
        bitmap_image image(w, h);

        float *results;
        cudaMallocManaged(&results, w * h * sizeof(float));

        int blockSize = 256;
        int numBlocks = (w + blockSize - 1) / blockSize;
        cudaCalc<<<numBlocks, blockSize>>>(w, h, xpos, ypos, z, results);

        cudaDeviceSynchronize();

        for(int x = 0; x < w; x++){
          for (int y = 0; y < h; y++){
            float value = results[x * w + y];
            image.set_pixel(x, y, (int)value, (int)value, (int)value);
          }
        }

        cudaFree(results);

        std::cout << "Finished rendering... Saving..." << std::endl;
        image.save_image("img.bmp");

        z = z * 1.1f;
    }
}