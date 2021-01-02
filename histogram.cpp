#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

typedef struct
{
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t align;
} RGB;

typedef struct
{
    bool type;
    uint32_t size;
    uint32_t height;
    uint32_t weight;
    RGB *data;
} Image;


cl_program load_program(cl_context context, const char *file_name) {
  // var
  size_t length;
  vector<char> data;
  ifstream infile(file_name, ios_base::in);
  
  // seekg is to set the position of the end of the file to tell the file length
  infile.seekg(0, ios_base::end);
  
  // Get the file length(in bytes)                                                       
  length = infile.tellg();       
  
  // seekg is to set the position of the beginning of the reading
  infile.seekg(0, ios_base::beg);

  data = vector<char>(length + 1);
  infile.read(&data[0], length);
  data[length] = 0;

  const char *source = &data[0];
  cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);

  if (program == 0 || clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS)
    return 0;

  infile.close();

  return program;
}

Image *readbmp(const char *filename)
{
    std::ifstream bmp(filename, std::ios::binary);
    char header[54];
    bmp.read(header, 54);
    uint32_t size = *(int *)&header[2];
    uint32_t offset = *(int *)&header[10];
    uint32_t w = *(int *)&header[18];
    uint32_t h = *(int *)&header[22];
    uint16_t depth = *(uint16_t *)&header[28];
    if (depth != 24 && depth != 32)
    {
        printf("we don't support depth with %d\n", depth);
        exit(0);
    }
    bmp.seekg(offset, bmp.beg);

    Image *ret = new Image();
    ret->type = 1;
    ret->height = h;
    ret->weight = w;
    ret->size = w * h;
    ret->data = new RGB[w * h]{};
    
    // Read a bmp Image via reading the size of the image and read (depth / 8) bytes at once time
    for (int i = 0; i < ret->size; i++)
    {
        bmp.read((char *)&ret->data[i], depth / 8);
    }
    return ret;
}

int writebmp(const char *filename, Image *img)
{

    uint8_t header[54] = {
        0x42,        // identity : B
        0x4d,        // identity : M
        0, 0, 0, 0,  // file size
        0, 0,        // reserved1
        0, 0,        // reserved2
        54, 0, 0, 0, // RGB data offset
        40, 0, 0, 0, // struct BITMAPINFOHEADER size
        0, 0, 0, 0,  // bmp width
        0, 0, 0, 0,  // bmp height
        1, 0,        // planes
        32, 0,       // bit per pixel
        0, 0, 0, 0,  // compression
        0, 0, 0, 0,  // data size
        0, 0, 0, 0,  // h resolution
        0, 0, 0, 0,  // v resolution
        0, 0, 0, 0,  // used colors
        0, 0, 0, 0   // important colors
    };

    // file size
    uint32_t file_size = img->size * 4 + 54;
    header[2] = (unsigned char)(file_size & 0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    // width
    uint32_t width = img->weight;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    // height
    uint32_t height = img->height;
    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;

    std::ofstream fout;
    fout.open(filename, std::ios::binary);
    fout.write((char *)header, 54);
    fout.write((char *)img->data, img->size * 4);
    fout.close();
}

int main(int argc, char *argv[])
{
    clock_t begin = clock();
    
    char *filename;
    if (argc >= 2)
    {
        int many_img = argc - 1;
        for (int i = 0; i < many_img; i++)
        {
            filename = argv[i + 1];
            Image *img = readbmp(filename);

            std::cout << img->weight << ":" << img->height << "\n";

            size_t max_items;
            size_t max_work[3];
            size_t local_work_size;
            size_t global_work_size;
            
            cl_int err;
            cl_uint num;
            
            cl_kernel kernel;
            cl_context context;
            cl_program program;
            cl_device_id device_id;
            cl_command_queue commands;
            cl_platform_id platform_id;
            
            cl_mem input;
            cl_mem output;
            
            unsigned int buffer;
            unsigned int total_tasks;
            unsigned int *image = NULL;
            unsigned int task_per_thread;
            unsigned int results[256 * 3];
            
            total_tasks = img->weight * img->height;
            image = new unsigned int[total_tasks * 3];
            
            for(int i = 0; i < total_tasks*3; i+=3){
              image[i] = img->data[i/3].R;
              image[i+1] = img->data[i/3].G;
              image[i+2] = img->data[i/3].B;
            }
            
            clGetPlatformIDs(1, &platform_id, NULL);
            
            err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
            if (err != CL_SUCCESS) {
              printf("clGetDeviceIDs(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work), &max_work, NULL);
            if (err != CL_SUCCESS) {
              printf("clGetDeviceInfo(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            max_items = max_work[0] * max_work[1] * max_work[2];
            
            task_per_thread = total_tasks / max_items + 1;
            
            context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
            if (!context) {
              printf("clCreateContext(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            commands = clCreateCommandQueue(context, device_id, 0, &err);
            if (!commands) {
              printf("clCreateCommandQueue(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            program = load_program(context, "histogram.cl");
            if (!program) {
              printf("load_program(): %d\n", program);
              return EXIT_FAILURE;
            }

            err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (err != CL_SUCCESS) {
              printf("clBuildProgram(): %d\n", err);
              return EXIT_FAILURE;
            }
          
            kernel = clCreateKernel(program, "histogram", &err);
            if (!kernel || err != CL_SUCCESS) {
              printf("clCreateKernel(): %d, %d\n", kernel, err);
              return EXIT_FAILURE;
            }
            
            input = clCreateBuffer(context, CL_MEM_READ_ONLY, total_tasks * 3 * sizeof(unsigned int), NULL, NULL);
            output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * 3 * sizeof(unsigned int), NULL, NULL);
            if (!input || !output) {
              printf("clCreateBuffer()\n");
              return EXIT_FAILURE;
            }
            
            err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, total_tasks * 3 * sizeof(unsigned int), image, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
              printf("clEnqueueWriteBuffer(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            err = 0;
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
            err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
            err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &total_tasks);
            err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &task_per_thread);
            if (err != CL_SUCCESS) {
              printf("clSetKernelArg(): %d\n", err);
              return EXIT_FAILURE;
            }
          
            err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL);
            if (err != CL_SUCCESS) {
              printf("clGetKernelWorkGroupInfo(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            global_work_size = max_items;
            err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
              printf("clEnqueueNDRangeKernel(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            clFinish(commands);

            err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, 256 * 3 * sizeof(unsigned int), results, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
              printf("clEnqueueReadBuffer(): %d\n", err);
              return EXIT_FAILURE;
            }
            
            clReleaseMemObject(input);
            clReleaseMemObject(output);
            clReleaseProgram(program);
            clReleaseKernel(kernel);
            clReleaseCommandQueue(commands);
            clReleaseContext(context);
          
            delete [] image;
            
            int max = 0;
            for(int i=0;i<256*3;i++){
                max = results[i] > max ? results[i] : max;
            }

            Image *ret = new Image();
            ret->type = 1;
            ret->height = 256;
            ret->weight = 256;
            ret->size = 256 * 256;
            ret->data = new RGB[256 * 256];
            
            for(int i=0;i<ret->height;i++){
              for(int j=0;j<ret->weight;j++){
                ret->data[i + ret->weight * j].R = 0;
                ret->data[i + ret->weight * j].G = 0;
                ret->data[i + ret->weight * j].B = 0;
              }
            }
            
            uint32_t R[256];
            uint32_t G[256];
            uint32_t B[256];
            
            for(int i=0;i<256;i++){
                R[i] = results[i];
                G[i] = results[i + 256];
                B[i] = results[i + 2*256];
            }
            
            for(int i=0;i<ret->height;i++){
                for(int j=0;j<256;j++){
                    if(R[j]*256/max > i)
                        ret->data[256*i+j].R = 255;
                    if(G[j]*256/max > i)
                        ret->data[256*i+j].G = 255;
                    if(B[j]*256/max > i)
                        ret->data[256*i+j].B = 255;
                }
            }

            std::string newfile = "hist_" + std::string(filename); 
            writebmp(newfile.c_str(), ret);
        }
    }else{
        printf("Usage: ./hist <img.bmp> [img2.bmp ...]\n");
    }
    
    clock_t end = clock();
    printf("Time Spend : %.5f\n\n", (double)(end - begin) / CLOCKS_PER_SEC);
    
    return 0;
}