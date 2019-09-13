## Overview
In today's exercise, we will continue our practice with CUDA for GPU programming.
We will focus on the types of memory available on a GPU system, and the mechanisms you use to move data between these types of memory.

## Part A: Host and Global Memory
The most basic question you have to ask about a region of memory in a GPU system is *"is this memory accessible on the GPU?"*
Host memory, which we've just called "memory" up to this point in the class, is the usual location for storing variables, code, constants, and any memory returned from `malloc`.

GPUs are separate devices, and generally cannot access host memory;
instead, they have their own large regions of memory called *global memory*.
If you declare global variables such as `int numbers[128]` you are asking for space in host memory to hold 128 integers.
Similarly, `__device__ int numbers[128]` requests space on the GPU's global memory to hold 128 integers.
While this is sometimes useful, we often use global memory to hold dynamically-allocated memory, returned from `cudaMalloc`.
Just as `malloc` returns a pointer to a block of memory usable on the host, `cudaMalloc` returns a pointer to a block of memory usable on the GPU.
You can use `cudaMemcpy` with the `cudaMemcpyHostToDevice` option as the fourth parameter to move memory from the host to the device, or pass `cudaMemcpyDeviceToHost` as the third parameter to move memory from the GPU back to host memory.

***Warning: pointers to GPU memory are not usable on the host, and vice versa!***
If a pointer was returned from `cudaMalloc`, it is only acceptable to dereference or index from that pointer on the GPU.
If a pointer was resturned from `malloc`, it is only acceptable to dereference or index from that pointer on the host.
You can pass values directly to kernels, but any time you pass a pointer you must be careful to check that the pointer points to GPU memory.

You'll often have both host and GPU versions of arrays you move back and forth, so it's a good idea to use a naming scheme like `int* cpu_numbers` and `int* gpu_numbers` to keep track of which is which.

Moving memory back and forth between the host and device is usually the bare minimum required to write functioning GPU code.
Complete the exercises below to practice basic data movement.

### Exercises
1. Write a simple CUDA program to create an array of 8388608 (which is 2^23) `double` values in the GPU's global memory using `cudaMalloc`. Initialize the array so index zero holds the value zero, index one holds the value one, index two holds the value two, and so on. *Hint: You will need to initialize the values on the host first and copy them into GPU memory.*

2. Now write a simple kernel, `halveValues(double* numbers, int length)`, that replaces each value in the array with half its original value. Each thread should perform one division operation. Use a block size of 64 threads, and compute the appropriate number of blocks given the array size. Wait for the kernel to compute with the `cudaDeviceSynchronize` function and copy the values back from the GPU.

3. Now modify your program from part 2 so it copies to the GPU, halves values, and copies back inside of a loop. How long does the program take to run one, two, and three iterations of this loop? You can measure the runtime of the program using the `time` command on the shell. You should discard the first run of a program after it has been compiled, since both CUDA and the OS do some additional work on the first run.

4. You may have already realized that copying to and from the GPU is not necessary between runs of the `halveValues` kernel, since the values are already on the GPU and are not needed on the host until the end of the program. Modify your program so there are only two copies: once before the loop to copy the values to the GPU, and once after the loop to copy the values back to the host. How long does this version take to run with one, two and three iterations of the loop? What does this tell you about the time it takes to invoke a kernel versus the time spent copying data back and forth?

## Part B: Managed Memory
You have probably realized by now that copying to and from the GPU is fairly costly, and keeping track of where the most recent version of a value currently resides can be complex.
This is often a major source of bugs for CUDA programmers.
Luckily, CUDA now has a feature that makes this easier called *managed memory*.

Managed memory in CUDA is memory that is allocated in such a way that the pointer works on both the GPU and the host.
To allocate managed memory, call the function `cudaMallocManaged(void** ptr, size_t sz)`.
You can read and write this memory on the host, then pass it to a kernel.
The memory will be copied over to the GPU as needed, and copied back if the memory is modified on the GPU and later accessed on the host.

When you use managed memory, you are free to acces values anywhere you like and CUDA will take care of tranferring the data for you.
However, the cost of moving data back and forth is the same;
you still need to worry about the performance cost of moving data around, and design your computations in a way that minimizes the number of transfers.

### Exercises

1. Make a copy of your halving kernel program from the previous part, and update it to use CUDA managed memory. How do the runtimes with one, two, and three calls to the kernel compare to the two versions you wrote for part A's exercises? You should find that CUDA's managed memory results in performance closer to the version of the program that minimizes transfers.

2. Now modify your program so each time the kernel returns, the host doubles each value in the array. What effect do you expect this change to have on the program's performance? Run it and measure the runtime to confirm your predictions.

## Part C: Shared Memory
GPU memory is often a bit faster than host memory, but because of their data-parallel design GPUs need quite a bit more data to keep them busy when compared to a normal processor.
In addition to a GPU's global memory, they also have *local* and *shared* memory regions that are faster, but limited in how they can be used.
Local memory is where a thread's local variables go, and you generally use it without thinking about it.

Shared memory is more interesting;
when you declare a local variable in your kernel with the `__shared__` type qualifier (e.g. `__shared__ int numbers[8]`) you get a region of memory than all threads in the current block can access.
Shared memory is good for two important things.
First, shared memory is much faster than global memory and can be used to hold values you will use repeatedly.
Second, shared memory allows you to perform some operations using intermediate values computed inside a block of threads.
Complete the exercises below to practice using shared memory for both purposes.

### Exercises
1. Start by writing a simple CUDA program that computes [finite differences](https://en.wikipedia.org/wiki/Finite_difference_method) across a sequence of 128 integers using a block size of 32 threads. This fancy-sounding method just means that you subtract the value at index i from the value at index i+1 to approximate the derivative. Populate the array so it represents the function $$y=x^2$$, or in C, `y[x] = x * x`. Keep in mind that your output will have one less element than the input. Run your kernel and verify that the output of the kernel is close to the function $$y=2x$$ (e.g. `y[x] = 2 * x`), the derivative of our original function. The slope should be correct, but you may be off by a small constant offset; why might that happen?

2. If you think back to the patterns for parallel computation we discussed earlier, the finite differences kernel is a good example of a *stencil*.
   Stencils work much like maps, but the kernel can look anywhere in the neighborhood of the value being mapped rather than just at the one value assigned to the mapper.
   That means there will be a lot of accesses to global memory, and neighboring stencil threads will access some of the same values.
   We can speed this process up by first copying values into `__shared__` memory.
   Add a variable `__shared__ int local_values[33]` to your kernel and have the threads in a given block copy values from global memory into this shared array.
   You should do the copying in parallel so each thread copies one value, and then choose a thread to copy the 33rd value (if there is one in the source array).
   Now modify the rest of your finite differences kernel so it computes differences by reading the shared array instead of global memory.
   Verify that your kernel produces the same output.
   If you are curious, you can try a larger input and compare running times between the two versions of the kernel;
   the one that uses shared memory should be faster for large inputs.

3. Another use for shared memory is to actually share values between threads in a block.
   To prepare for the next exercise, write a kernel that computes the sum of squares of all the values in an array.
   The simplest way to do this is to read each index from global memory, square it, and then add it to a single global output value using `atomicAdd` to prevent our updates from racing.
   Test your kernel on an array of at least 100 integers and verify the output.

4. Remember that accesses to global memory are slow;
   atomic accesses are even slower!
   Instead of making one atomic update per index in the input array, we are going to use shared memory to *coalesce* updates.
   Modify your kernel so each thread computes the square of its input value and stores that in a `__shared__` array.
   Then, call the `__syncthreads()` function to force all threads in the block to wait (this is a barrier in CUDA).
   After that point, have *one* thread sum the values in the shared array and make a single update to the global sum.
   *Does this coalesced update have to use `atomicAdd`? Why or why not?*

