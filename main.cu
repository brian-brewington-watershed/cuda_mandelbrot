#include <iostream>
#include "bitmap_image.hpp"
#include <math.h>

const int MAX_ITERATION = 255;

__global__
void cudaCalc(int w, int h, double mx, double my, double zoom, double *results){
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;

  double scaledX = ((double)px - w / 2) / (w / 2) / zoom + mx;
  double scaledY = ((double)py - h / 2) / (h / 2) / zoom + my;
    
  double x = 0;
  double y = 0;
  
  double it = 0;
  
  while (x * x + y * y < 4 && it < MAX_ITERATION)
  {
      double xTemp = x * x - y * y + scaledX;
      y = 2 * x * y + scaledY;
      x = xTemp;
  
      it++;
  }

  results[px * w + py] = it;
}

int main(void){
    int w = 500;
    int h = 500;

    double xpos =  -0.7463f;
    double ypos = 0.1102f;

    double z = (int)1;

    while(true)
    {
        bitmap_image image(w, h);

        double *results;
        cudaMallocManaged(&results, w * h * sizeof(double));
        std::cout << "Allocated memory" << std::endl;

        dim3 block(32,32);
        dim3 grid;
        grid.x = (w + block.x - 1)/block.x;
        grid.y = (h + block.y - 1)/block.y;
        cudaCalc<<<grid, block>>>(w, h, xpos, ypos, z, results);
        std::cout << "Started task" << std::endl;

        cudaDeviceSynchronize();

        std::cout << "Finished task, constructing image" << std::endl;

        for(int x = 0; x < w; x++){
          for (int y = 0; y < h; y++){
            double value = results[x * w + y];
            image.set_pixel(x, y, (int)value, (int)value, (int)value);
          }
        }

        std::cout << "Finished rendering... Saving..." << std::endl;
        image.save_image("img.bmp");

        std::cout << "Finished image, freeing shared memory" << std::endl;
        cudaFree(results);

        z = z * 1.1f;
    }
}