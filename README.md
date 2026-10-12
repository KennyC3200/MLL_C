# README
A machine learning library that trains an image classification model for handwritten digit recognition. The model is trained on 60,000 MNIST images and achieved 92.4% accuracy.

## How to run
1. Install a C compiler (e.g. [GCC via MinGW-w64](https://www.mingw-w64.org/) on Windows, or `gcc`/`clang` on Linux/macOS).
2. Open a terminal in the project folder. Make sure `train_images.mat`, `train_labels.mat`, `test_images.mat` and `test_labels.mat` are in the same folder.
3. Compile:
   ```
   gcc main.c -o main -O2 -lm
   ```
4. Run:
   ```
   ./main        # Linux/macOS
   main.exe      # Windows
   ```