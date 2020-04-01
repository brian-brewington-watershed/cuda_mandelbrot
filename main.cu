#include <iostream>
#include "bitmap_image.hpp"
#include <math.h>
#include "GL/freeglut.h"

const int MAX_ITERATION = 255;

int width = 1000;
int height = 1000;

GLuint bufferID;

__global__ void cudaCalc(int w, int h, double mx, double my, double zoom, double *results)
{
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

void draw()
{
  

  glutSwapBuffers();
}

void enable2D(int width, int height){
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0f, width, 0.0f, height, 0.0f, 1.0f);
  glLoadIdentity();
}

int main(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(width, height);
  glutCreateWindow("This is a test");

  //define callbacks
  glutDisplayFunc(draw);

  //setup scene for 2D and set draw color of everything to white
  enable2D(width, height);
  glColor3f(1.0f, 1.0f, 1.0f);

  glutMainLoop();
  return 0;
}
