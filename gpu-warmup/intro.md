This lab will give you a quick introduction to the basics of GPU programming.
We'll start using CUDA C, Nvidia's extension to the C programming language that allows you to write code that executes on the GPU instead of your computer's CPU.
Material for this lab is based on [a presentation by Jason Sanders at the GPU Technology Conference in 2010](http://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf).

## What is a GPU?
Graphics Processing Units (GPUs) were originally developed to perform computations to render and display graphics, but it turns out the computational model that GPUs offer is useful for much more than graphics code.
GPUs aren't the only example, but they are by far the most common example of a *co-processor*, a piece of hardware where you can perform computation in addition to your computer's CPU.
This shift has been quite dramatic over the last decade;
many GPUs are explicitly designed for parallel computation rather than graphics code, and most of the [Top 500 Supercomputers](http://www.top500.org/lists/2017/11/) use GPUs or other co-processors.

At a high level, a GPU is a data-parallel processor.
This is quite different from your CPU, which primarily uses task parallelism.
Recall that task parallel computation allows you to run different kinds of computational tasks on separate processors, which operate on mostly-independent data.
In contrast, data-parallel computation performs roughly the same operation (called a *kernel*) on each element of a data array.
GPUs are much better suited to this second model;
they have some support for coordination between parallel operations, but the more you use this functionality the slower your program will be.

CUDA is an extension to the C programming language that gives you the ability to specify whether a function executes on the host (CPU) or device (GPU), and to invoke operations on the device from the host.
Details are available in the [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/) from Nvidia.
There is a alternate programming model called OpenCL that supports a wider variety of devices, but OpenCL is a bit lower-level and can be harder to use.

## Introduction to CUDA
We'll start by looking at this CUDA *"Hello World"* program, which you should put into a file named `hello.cu`.

```c
#include <stdio.h>

__global__ void kernel() {
  printf("Hello world!\n");
}

int main() {
  kernel<<<1,1>>>();
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  return 0;
}
```

As you can see, CUDA code resembles C code for the most part. There are a few key differences though:

 - The `__global__` keyword on line 3 tells CUDA that this is a function that runs on the device (GPU), but can be called from the host (CPU). The alternatives are `__device__`, which is a function that can only be called on the device (usually a helper function) or `__host__`, the default, which means the function can only be called and run on the CPU.

 - Line 8 looks a bit like a function call, but the extra `<<<` and `>>>` markers tell CUDA how many *blocks* and *threads* to use to execute the `kernel` function (more on these later).

 - The call to `cudaDeviceSynchronize()` on line 9 is normal C code that waits for the call to `kernel` to complete before returning from the `main` function. This function returns `cudaSuccess` when the kernel works correctly, otherwise it returns an error code.
 
 - The error handling code gets the return code from the last CUDA procedure, turns it into a string, and prints it out. This is similar to how we've used `perror` in the past, although it's a bit less convenient.

To compile this code, use the `nvcc` compiler.
This compiler may not be in your path, so try running it with no arguments first.
If you receive an error, you will need to add the compiler to your path.
You can do this with a shell one-liner:

```shell
$ echo "export PATH=\"\$PATH:/usr/local/cuda/bin\"" >> ~/.bashrc
$ source ~/.bashrc
```

This tells your shell to search for executables in the CUDA installation directory.

Once you have verified that `nvcc` is available on your machine, run the following command to compile your code for the GPU:

```shell
$ nvcc -o hello hello.cu
```

This produces an executable called `hello` that contains both the CPU and GPU portions of the program. Run the program as you normally would:

```shell
$ ./hello
Hello world!
```

While this may seem like a fairly typical program, the output actually comes from the GPU part of the program rather than the part that executes on the CPU.
Printing isn't very interesting (or efficient) on the GPU, so we'll look at more reasonable computations.
But first, we have to look at how GPU programs create threads.

### Blocks and Threads
The key part of the code above is line 8, which invokes the `kernel` function with one block and one thread.
If we're going to take full advantage of the available GPU hardware, we need to use parallelism, but there are two ways to do this: blocks and threads.
Blocks are collections of threads that are scheduled together, and a thread is much like a thread on your CPU (at least in terms of the programming model).
If a program has multiple threads and blocks, CUDA will try to schedule as many blocks as possible at one time, then run new blocks once those complete.
Threads within a block are always scheduled together.

You can increase the concurrency of `hello.cu` by changing the number of blocks, threads, or both. What do you think the program output will be for the following calls?

 - `kernel<<<4,1>>>();`, four blocks of one thread each
 - `kernel<<<1,8>>>();`, one block of eight threads
 - `kernel<<<2,5>>>();`, two blocks of five threads
 
As you can see, increasing blocks and threads both increase the number of times the `kernel` function is called.
While this is a nice simple example, it's not particularly useful to print the same message every time.
GPUs are well-suited to programs that operate on arrays (or *vectors* in the mathematical sense), and we'd often like each invocation of our kernel to handle a specific element of the array.

The code below shows a variant of the hello world program that prints out block and thread IDs.

```c
#include <stdio.h>

__global__ void kernel() {
  printf("Hello from block %d thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
  kernel<<<4,6>>>();
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  return 0;
}
```

Compile this program with `nvcc` as before and run it.

#### Exercises
Before moving on, answer the following questions.

1. If you change the number of threads, how does that impact the order that messages are printed?
2. If you change the number of blocks, how does that impact the order that messages are printed?
3. What can you conclude about how blocks and threads are run on the GPU?

### SAXPY

One of the simplest useful array operations is called SAXPY, short for *single-precision A times X plus Y*.
For those of you who remember basic algebra, this is computing the value of a function of the form y=mx+b using `float` types rather than `double`s.
Each thread should read a value in the X array, multiply it by a constant A, add a value from the Y array, and then put it somewhere.
Usually this process is used to progressively update values in order to approximate differential equations, so we store the result into the Y input.

To make this work with CUDA, we'll need to pass array parameters to our kernel function, then tell each invocation of the kernel deal with a different element of the arrays.
You can do this using the values `threadIdx.x` and `blockIdx.x`, which are visible in a kernel executing on the GPU.
These tell you the thread and block ID of the currently running thread.
Threads are numbered from zero up to block size minus one, and blocks are numbered from zero to the number of blocks minus one.
You can also use `blockDim.x` to get the number of threads in the current block.

The code below shows a partial SAXPY implementation:

```c
#include <stdint.h>
#include <stdio.h>

#define N 32
#define THREADS_PER_BLOCK 32

__global__ void saxpy(float a, float* x, float* y) {
  // Which index of the array should this thread use?
  size_t index = _______;
  y[index] = a * x[index] + y[index];
}

int main() {
  // Allocate arrays for X and Y on the CPU
  float* cpu_x = (float*)malloc(sizeof(float) * N);
  float* cpu_y = (float*)malloc(sizeof(float) * N);
  
  // Initialize X and Y
  int i;
  for(i=0; i<N; i++) {
    cpu_x[i] = (float)i;
    cpu_y[i] = 0.0;
  }
  
  // Allocate space for X and Y on the GPU
  float* gpu_x;
  float* gpu_y;
  
  if(cudaMalloc(&gpu_x, sizeof(float) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }
  
  if(cudaMalloc(&gpu_y, sizeof(float) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate Y array on GPU\n");
    exit(2);
  }
  
  // Copy the host X and Y arrays to the device X and Y arrays
  if(cudaMemcpy(gpu_x, cpu_x, sizeof(float) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X to the GPU\n");
  }
  
  if(cudaMemcpy(gpu_y, cpu_y, sizeof(float) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y to the GPU\n");
  }

  // How many blocks should be run, rounding up to include all threads?
  size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  // Run the saxpy kernel
  saxpy<<<blocks, THREADS_PER_BLOCK>>>(0.5, gpu_x, gpu_y);
  
  // Wait for the kernel to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  
  // Copy values from the GPU back to the CPU
  if(cudaMemcpy(cpu_y, gpu_y, sizeof(float) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y from the GPU\n");
  }
  
  for(i=0; i<N; i++) {
    printf("%d: %f\n", i, cpu_y[i]);
  }
  
  cudaFree(gpu_x);
  cudaFree(gpu_y);
  free(cpu_x);
  free(cpu_y);
  
  return 0;
}
```

This program is quite a bit longer, but you can ignore the contents of `main` for now.
In the kernel function, called `saxpy` this time, you can see we're just accessing one index of the two arrays.
The code to compute *which* index is left blank;
answer the following questions and update the code:

 - If there is just one block, what index should each run use?
 - What if there are multiple blocks?
 - If you run the program with `N` of 48 and a block size of 32 you will have two blocks of 32 threads, but not all of these threads have elements of the array to process. How can you prevent the kernel from reading or writing off the end of the array?

Raise your hand once you have updated your `saxpy` kernel and I will check to make sure your work is correct.

Now it's time to dig into the `main` function.
Lines 14--24 deal with memory allocation and initialization;
this should all be familiar.
But memory on your CPU isn't the same as memory on the GPU.
Lines 25--37 allocate memory on the GPU to old the values for X and Y, and lines 40--46 copy data from the CPU to the GPU.
After this, the code computes the number of blocks that should be run, making sure to schedule enough that there are at least as many total threads as there are elements in the array.
Finally, line 52 runs the `saxpy` kernel, line 55 waits for the kernel to finish, and lines 58--64 copy the memory back from the GPU and print out the results.

What do you expect the output of this program to be for a range of `N` and `THREADS_PER_BLOCK` values?
Compile the program with `nvcc` and run it to test your predictions.
Make sure to test cases where `N` does not evenly divide by the number of threads in each block.
Once you think your program is running correctly, raise your hand and I will check the results.

### Why do we need both blocks and threads?
So far we haven't seen a compelling reason to use both blocks and threads;
the GPU will run threads in parallel either way, and the fact that thread within a block are scheduled together is not useful when we just wait for all threads to finish.
The distinction between threads and blocks becomes important when we care about *interactions between threads*.
Synchronization allows you to coordinate threads executing concurrently, but only within a single block.
Threads that are running in separate blocks cannot synchronize.

One example of a task that requires coordination is a *dot product*.
The dot product of two vectors is the sum of the pairwise products of all elements in the two vectors.
The following kernel code computes a dot product, but is missing critical synchronization:

```c
...

__global__ void dotproduct(float* x, float* y, float* result) {
  // Compute the index this thread should use to access elements
  size_t index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  
  // Create space for a shared array that all threads in this block will use to store pairwise products
  __shared__ float temp[THREADS_PER_BLOCK];
  
  // Compute pairwise products
  temp[threadIdx.x] = x[index] * y[index];
  
  // The thread with index zero will sum up the values in temp
  if(threadIdx.x == 0) {
    float sum = 0;
    int i;
    for(i=0; i<THREADS_PER_BLOCK; i++) {
      sum += temp[i];
    }
    
    // Add the sum for this block to the result
    *result += sum;
  }
}

...
```

This example code uses a new CUDA feature, the `__shared__` type annotation.
This means the memory used for this variable is shared between all threads used within the current block.
Each thread computes one pairwise product of its elements in `x` and `y`, then stores the result into its element of `temp`.
Then, the thread with index zero loops over the list of pairwise products and adds them up.
Finally, the sum for this block is added to the global result, which may include result from other blocks that have computed sums of pairwise products for subsets of the input arrays.

Using the SAXPY code as a template, write a harness that launches this kernel on a pair of arrays of length 64 using a single block.
What happens when you run the program? *Hint: you will not get the right answer.*
Do you see what is going wrong?

The issue with this code is that not all threads are guaranteed to complete before thread zero computes the sum for the block.
To make sure all values in `temp` have been computed before moving on to the sum stage, you can add a call to `__syncthreads()` in the kernel.
This will force all threads to wait until every thread in the block has completed the pairwise product portion of the code.

This code also doesn't properly handle values of `N` that aren't evenly divisible by the block size.
Update the code so it will not read or write out of bounds indices in `x` and `y` with `N` set to 48 and `THREADS_PER_BLOCK` set to 64.

If you move to input sizes that require multiple blocks, there is a new race condition.
Can you find it?
Raise your hand once you think you've found the issue so I can check.

Unfortunately, we can't use normal synchronization to fix this problem.
Different blocks run at different times, and some blocks may not start until the current one finishes.
That means if we try to wait for all blocks to finish we may be waiting forever.
Instead, we can use *atomic operations* to sidestep this issue.
The `atomicAdd` function takes two parameters, a pointer to a memory location that should be written to and a value to add to that location.
Use `atomicAdd` to fix the dot product implementation and test it with values of `N` that require multiple blocks.
You can find documentation for all of CUDA's atomic operations [here](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd).

### Exercises: Day One
1. Implement and test a kernel that computes X to the power of Y, where X and Y are vectors of `float`s. You can find a refernce for floating point operations in CUDA [here](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE).
2. Implement and test a kernel that reverses an array that may span multiple blocks.
3. Implement and test a kernel that updates each value in a vector to the average of its value and it's immediately-adjacent neighbors. For values on the edge, just average the value and its one neighbor. What do you expect this kernel to do if you run it repeatedly on an array of random values?

#### If you have extra time
Implement a **reducer** for a single array: one that sums all elements of an array using as much parallelism as possible.
You may find it easiest to implement a solution that performs the reduction across multiple kernel invocations;
that is perfectly fine.
Keep in mind that addition is associative and commutative: you can compute pairwise sums of neighboring elements, add neighboring pairs of those sums, and so on until you have a single value.
