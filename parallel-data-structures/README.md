## Overview
In this lab, you will implement three different *abstract data types* that support concurrent access by multiple threads and write tests to verify that your data structures work correctly under concurrent modification. You will implement a [stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type)), [queue](https://en.wikipedia.org/wiki/Queue_(abstract_data_type)),
and [associative array](https://en.wikipedia.org/wiki/Associative_array).
You will then test these implementations using the [Google Test](https://github.com/google/googletest) framework.

The first time you run `make` in the starter code it will download the Google Test framework, so make sure you have an internet connection the first time you build the code.
This lab should run correctly on both Linux and macOS machines.

## Questions & Answers
Do we need to write one test case per invariant?
: Not necessarily. You may find it's easier to test multiple invariants at once.

Do we need to grow our hash table for part C?
: No. If you use a hash table, I recommend chaining rather than linear probing, otherwise you will have a hard upper limit on capacity.

How do we test whether some accesses can run concurrently?
: You probably can't do this easily. Look over your code and convince yourself that it works.

How do we make threads in our test files?
: The `TEST(Something, SomethingElse) {` line starts a function that Google Test will run. You can call `pthread_create` from this function just like you would from `main`.
  
Can we run `CHECK_EQ` and other checking functions in threads with Google Test?
: No, it appears that this is not supported. Your threads will need to report back to the main thread what they observed so you can run your checks there.
  
Do you have any hints for testing with threads? It seems like a lot of behavior is unpredictable because we don't know the order our threads will run in.
: I recommend creating threads to perform a series of operations in parallel, then do the final checking after the threads have exited. This will require that you set up your test cases in a way that makes it possible to check them later; for example, have one thread push odd numbers in ascending order and another thread push even numbers in ascending order. After this finishes you can make sure you find the right number of values on the stack, the odds are in descending order, and the evens are in descending order.

## Part A: Stack
As you likely remember from CSC 161, a stack is a simple data structure that supports two operations: adding items to the top, and removing items from the top. 
You should complete the implementation for a stack data structure in `stack.cc` and `stack.hh`.
You may notice these files have slightly different extensions than you are used to;
that's becuase these will actually be compiled as C++ instead of C so the implementation is compatible with Google's test framework.
You can still write completely normal C, so there's no need for you to learn anything about C++ for this lab.

Implement a thread-safe stack with the following interface:

**`my_stack_t`**  
This type holds all of the information you need to access a stack. This struct is defined in `stack.hh`. You will need to add fields to this struct regardless of how you choose to implement your stack.

**`void stack_init(my_stack_t* stack)`**  
This function takes a pointer to memory that can hold a `my_stack_t` and initializes it to an empty struct. Complete the implementation of this function in `stack.cc`.

**`void stack_destroy(my_stack_t* stack)`**  
This function takes a `my_stack_t` and destroys it. This should free any memory allocated by your code with pointers inside the stack, but not the `stack` parameter itself, since this memory was not allocated by your code. The program using your data structure should not use the stack passed to this function without re-initializing it. Complete the implementation of this function in `stack.cc`.

**`void stack_push(my_stack_t* stack, int element)`**  
Push an integer onto the stack. The provided `my_stack_t*` must have been initialized with `stack_init`, and must not have been passed to `stack_destroy`. Complete the implementation of this function in `stack.cc`.

**`bool stack_empty(my_stack_t* stack)`**  
Check if a stack is empty. Complete the implementation of this function in `stack.cc`.

**`int stack_pop(my_stack_t* stack)`**  
Pop an integer from the stack. The provided `my_stack_t*` must have been initialized with `stack_init`, must not have been passed to `stack_destroy`, and must not be empty. If the stack is empty, `stack_pop` should return `-1`. If the stack is not empty, the integer on the top of the stack should be removed from the stack and returned. Complete the implementation of this function in `stack.cc`.

### Requirements
You are free to choose any implementation method you like for your stack, but *you may not assume an upper limit on the number of elements that will be in the stack.* That means a static-size array will not work.

You must also make your stack reentrant; it must be possible to have two or more independent stacks that can be pushed and popped independently. This means there should be no global variables in your stack implementation. You will need to keep any locks or per-stack data inside the `stack_t` struct or a linked structure that appears inside your stack.

Multiple threads may access the same stack at the same time, so you will need to use synchronization inside your stack implementation to maintain the integrity of this data structure.
When we think about data structures, it is often helpful to write down *invariants*---statements that must always be true about the data structure.
Your synchronization for the stack data structure must maintain the following invariants:

Invariant 1
: For every value $$V$$ that has been pushed onto the stack $$p$$ times and returned by pop $$q$$ times, there must be *p-q* copies of this value on the stack. This only holds if $$p >= q$$.

Invariant 2
: No value should ever be returned by pop if it was not first passed to push by some thread.

Invariant 3
: If a thread pushes value *A* and then pushes value *B*, and no other thread pushes these specific values, *A* must not be popped from the stack before popping *B*.

Invariant 1 describes the basic requirements for most reasonable data structures: items that are added to the data structure but not removed should still be contained in the data structure. There cannot be a negative number of copies of a value on the stack, so this also implies that a value should never be returned from pop more times than it has been pushed.

Invariant 2 just says the stack is not allowed to invent new values that were never added to the stack. This is another reasonable requirement for most data structures.

Invariant 3 is specific to stacks;
it restricts the order in which pop may return values, based on the order these values are pushed.
If two values are pushed from one thread they should be popped in the reverse order.

This invariant says nothing about the order of pushes and pops carried out by threads running in parallel.
If two threads are pushing onto the stack simultaneously, both values should end up on the stack *in some order*.
If one thread pushes a value while the other thread pops a value, the popping thread could receive the old or new top value from the stack.

### Testing
You will use the Google Test framework to test your data structure implementations in this lab.
The file `stack-tests.cc` shows a simple example test.
You will need to add tests for each invariant to this file.
To run these tests, run `make` and then run the `stack-tests` executable.
Running this program will execute your tests and report the results on the terminal.

The provided test code uses `ASSERT_EQ` to require that two values are equal, and `ASSERT_NE` to require that two values are *not* equal.
When comparing strings, make sure to use `ASSERT_STREQ` and `ASSERT_STRNE`.
You can find simple test guidance on the Google Test [Primer](https://github.com/google/googletest/blob/master/googletest/docs/primer.md), and a complete list of comparisons on the [Advanced Guide](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md).

#### Testing Invariants 1 and 2
To test these invariants, create at least two threads that push and pop random values.
These threads should keep track of how many times they pushed each value (perhaps a random integer between 0 and 100) as well as the number of times they popped each value.
When these threads finish executing some number of pushes and pops, return the counts of pushed and popped values to the main thread.
The main thread can then pop the remaining values from the stack and make sure all the totals for pushed values match those of the popped values.
At this point you should make sure all the values popped were also pushed, which checks invariant 2.

#### Testing Invariant 3
This invariant has to do with pushing and popping order.
It may be difficult to validate the order of values popped from a stack if you allow unconstrained values to be pushed, but you can be clever about how you choose values to push.
Instead of pushing random values, you could create a thread that pushes even numbers in ascending order, and another thread that pushes odd values in ascending order.
When you pop these values (most likely from the main thread) you can simply keep track of the last even and odd values you popped from the stack;
if you ever see pop return a larger even or odd value than the last one you saw, the invariant has been violated.

To test this invariant, you should create two test cases:
one where pushes happen in parallel and pops happen in the main thread, and a second that pushes in the main thread and pops in parallel.
This ensures that both pushes and pops can proceed in parallel without unintended reorderings in the stack.

#### Testing Strategies
I have not included the testing strategies for parts B and C of the lab.
You will have to come up with these on your own, although they may be very similar to the tests for part A.
In general, constraining the sequence of inputs to a data structure to make it easy to validate outputs is a good approach that you should use whenever possible.

### Grading
Your grade for this part of the lab will depend on several factors:

1. Do your tests adequately check that the data structure invariants are preserved?
2. Is your data structure properly synchronized?
3. Is your code concise, clean, and well-documented?
4. Does your code build without warnings?
5. Does your code consistently check for error conditions in standard library calls?
6. Is your program free of memory leaks?

## Part B: Queue
Like a stack, a queue has a very simple interface. However, a queue returns elements in the order they were added to the queue, not the reverse. These data structures are often described with the acronyms LIFO (last in, first out) and FIFO (first in, first out) for stacks and queues, respectively.
Your queue should implement the following interface:

**`my_queue_t`**  
This type holds all of the information you need to access a queue. It should probably be defined as a struct. This struct is defined in `queue.hh`. You will need to add fields to this struct.

**`void queue_init(my_queue_t* queue)`**  
This function should initialize an empty queue in the space passed in with the `queue` parameter. Complete the implementation of this function in `queue.cc`.

**`void queue_destroy(my_queue_t* queue)`**  
This function takes a `my_queue_t*` and destroys it. This should free any allocated memory associated with the given queue. It is not safe to do *anything* with the `my_queue_t*` after it has been passed to this function. Complete the implementation of this function in `queue.cc`.

**`void queue_put(my_queue_t* queue, int element)`**  
This function takes a queue that has been initialized with `queue_create` (and has not been passed to `queue_destroy`) and an integer to add to the queue. It then adds `element` to the queue. Complete the implementation of this function in `queue.cc`.

**`bool queue_empty(my_queue_t* queue)`**  
This function returns true if the queue contains no items. Complete the implementation of this function in `queue.cc`.

**`int queue_take(my_queue_t* queue)`**  
This function takes a queue, removes the first element, and returns this element. If the queue is empty, the function should return `-1`. Complete the implementation of this function in `queue.cc`.

### Requirements
You are free to choose any implementation method you would like for your queue, provided you do not assume any upper bound on the number of elements in the queue.
As with your stack implementation, the queue must be reentrant: independent queue can exist simultaneously, so no globals please.

As with your stack, the queue may be accessed simultaneously by multiple threads.
Add synchronization to your queue to maintain the data structure's integrity.
You may want to refer to the reading in locked data structures for a clever implementation technique that allows `queue_put` and `queue_take` to run concurrently.

There is space at the top of `queue-tests.cc` for you to write invariants for your queue data structure.
Write down the invariants you believe capture the expected behavior for your queue.
You may find it helpful to refer to the invariants for the stack data structure when writing invariants for your queue.

### Testing
Use the Google Test framework to test your queue implementation.
Make sure your tests adequately check all of the invariants you specify.
Remember that thread interactions are difficult to control, so an improperly-synchronized data structure may pass all of its tests.
If you make your tests run longer, with enough concurrent accesses, you are more likely to catch an error.

The included test cases check the basic functionality of your queue.
Please leave these in place;
add additional test cases to check your invariants.

### Grading
Your grade for this part of the lab will depend on several factors:

1. Are your data structure invariants accurate and reasonably complete?
2. Do your tests adequately check that these invariants are preserved?
3. Is your data structure properly synchronized?
4. Is your code concise, clean, and well-documented?
5. Does your code build without warnings?
6. Does your code consistently check for error conditions in standard library calls?
7. Is your program free of memory leaks?

## Part C: Associative Array
One particularly useful abstract datatype is an *associative array*, also known as a *dictionary* in python and *map* in C++ and Java. An associative array works like an array with two major differences: the index into an associative array is not necessarily an integer (we will use strings for this lab), and the "array" does not have a fixed size. We will implement an associative array that stores integers, and uses strings as indices into the associative array.

To access an associative array, you may *get* the value with a given key, *set* a key to a particular value, or *remove* the entry for a given key.

To keep function names short, we will use the name "dict" for this datatype in the interface below. Your associative array should have the following interface:

**`my_dict_t`**  
This type holds all of the information required to access a dictionary. This struct is defined in `dict.hh`, but you will need to add fields to complete your implementation.

**`void dict_init(my_dict_t* dict)`**  
This function initializes a dictionary in the memory pointed to by `dict`. Complete the implementation of this function in `dict.cc`.

**`void dict_destroy(my_dict_t* dict)`**  
This function takes a `my_dict_t*` and destroys it. This should free any allocated memory associated with the dictionary. It is not safe to do *anything* with the `my_dict_t*` after it has been passed to this function. Complete the implementation of this function in `dict.cc`.

**`void dict_set(my_dict_t* dict, const char* key, int value)`**  
This function adds or updates an entry in `dict`. If there is already an entry for `key`, then its value will be updated. If no entry exists for `key`, a new one is added. Complete the implementation of this function in `dict.cc`.

**`bool dict_contains(my_dict_t* dict, const char* key)`**  
This function should return `true` if the dictionary contains a value with the given key. Complete the implementation of this function in `dict.cc`.

**`int dict_get(my_dict_t* dict, const char* key)`**  
This function looks up a key in `dict`. If there is an entry for `key`, the last value associated with this key is returned. If no matching entry exists, the function returns `-1`. Complete the implementation of this function in `dict.cc`.

**`void dict_remove(my_dist_t* dict, const char* key)`**  
This function removes any entry associated with `key`. If there is no entry for `key`, this function does nothing. After removing a key from the dictionary, `dict_contains` should return `false` for that key, even if it was set multiple times. Complete the implementation of this function in `dict.cc`.

### Requirements
You are free to choose any implementation method for your associative array, with some reservations.
You may not assume any upper bound on the number of elements that will be in a dictionary, your implementation must be re-entrant, and *you may not use a linear scan to implement the entire dictionary.*
Hash tables and binary search trees are both reasonable implementation strategies, but scanning through a linked list or array is not.
Make sure you document your design using comments in `dict.cc`, including details about your synchronization strategy.

As with the two previous data structures, you should add invariants to the top of `dict-tests.cc`.
The invariants for a dictionary will be more complex than those for stacks or queues. You will need to write tests to check these invariants when there are concurrent accesses to your dictionary, so keep that in mind as you write them.

Multiple threads may access the same dictionary concurrently, and you are required to allow concurrent accesses whenever possible; it should be possible for two calls to `get` to run in parallel, at least some of the time. The exact details of what kinds of accesses can happen in parallel will depend on your implementation. Make sure you answer the two questions about synchronization at the top of `dict-tests.cc` to explain how your synchronization allows and disallows certain types of concurrent accesses.

### Testing
Use Google Test to check the invariants you specified for your dictionary data structure. Remember that thread interactions are difficult to control, so an improperly-synchronized data structure may pass all of its tests. If you make your tests run longer, with enough concurrent accesses, you are more likely to catch an error.

The included test cases check the basic functionality of your dictionary.
Please leave these in place;
add additional test cases to check your invariants.

You do not need to test whether certain accesses will proceed in parallel or not. This is especially difficult to check with automated tests, so you will likely need to resort to careful code reading to verify that your synchronization scheme is working.

### Grading
Your grade for this part of the lab will depend on several factors:

1. Are your data structure invariants accurate and reasonably complete?
2. Do your tests adequately check that these invariants are preserved?
3. Is your description of allowed concurrent accesses to the data structure accurate?
4. Is your data structure properly synchronized?
5. Is your code concise, clean, and well-documented?
6. Does your code build without warnings?
7. Does your code consistently check for error conditions in standard library calls?
8. Is your program free of memory leaks?
9. Does your synchronization scheme allow concurrent reads without blocking?
10. Does your synchronization scheme allow concurrent writes *to different keys* without blocking?
