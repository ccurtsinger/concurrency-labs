## Overview
In this lab, you will implement a simple system for running a series of tasks within one process.
Tasks in this system will run separate functions, and your scheduler will need to switch between them.
Unlike the tasks we've seen in our discussion of scheduling so far, these tasks will have the ability to perform blocking operations.
At this point, your scheduler will need to stop running the current task and select a new one to run in the meaintime.
You will not use preemption;
your scheduler only switches tasks when the currently-running task blocks.
This is sometimes referred to as cooperative scheduling, since it relies on all the tasks to cooperate by blocking periodically.

While the system you implement could be used for other purposes, you will be implementing it specifically to support the game **Worm!**, a clone of the classic game *Snake*.

This game implementation uses several tasks:

1. A main task that starts up the game
2. A task to update the worm state in the game board
3. A task that redraws the board periodically
4. A task that reads input from the user and processes controls
5. A task that generates "apples" at random locations on the board
6. A task that updates existing "apples" by spinning them and removing them after some time

Each task runs in a loop that contains a blocking operation.
Your task is to implement a scheduling system that will run these tasks in round-robin fashion, switching between them as the currently-executing task blocks. We will spend some time at the start of class looking through the provided code and discussing the scheduler implementation.

## Questions & Answers
How should `task_readchar` work?
: If there's already a character to read, we could return it to the task immediately. However, if `getch()` returns `ERR`, no input is available. First, you need to record why the task is blocked (it's waiting for input), and then invoke your scheduler to choose a new task to run. Later, when the scheduler is choosing a task to run (at some point in the future) there will be an input character. The scheduler can then `swapcontext()` back to the task we blocked. 

## Scheduler System Details
You will be implementing what is known as *cooperative scheduling*.
This is a simple technique for implementing a scheduler without preemption.
Once a task is started it will run until it issues a blocking operation or exits.
In our case, tasks will run until they exit, wait for another task, sleep for a fixed amount of time, or wait for user input.
When you hit one of these points you should invoke the scheduler function to select and start another task to run.
As part of this process, you will need to check to see if any previously-blocked tasks can now unblock.
This could happen because they are waiting for tasks that have now completed, their sleep timer has elapsed, or user input is now available.
The scheduler API below should give a better sense of how this all works.

### API
`task_t`
: This is a type that users of the scheduler library use to uniquely identify tasks. In the starter code, this is just an `int`, which is then used as an index into an array of task information structures.

`void scheduler_init()`
: Programs should call this function before invoking any other scheduler functions. This gives you an opportunity to set up any bookkeeping data you need. Odds are you'll have to set things up to keep track of the program's main task.

`void task_create(task_t* handle, void (*fn)())`
: This function creates a new task to run the function passed in as `fn`, and writes an identifying handle for this new task to the first parameter. Your task functions must take no parameters and return void.

`void task_wait(task_t handle)`
: When a task calls this function, it is waiting for the task specified by `handle` to finish. You can think of this a bit like `wait` for processes, although we do not need to deal with exit statuses. The value in `handle` should have been set by a call to `task_create`.

`void task_sleep(size_t ms)`
: The calling task should block for `ms` milliseconds. At this point, you should schedule another task to run. You cannot simply call `sleep_ms` in this function, since that would block *all* task, not just the caller.

`int task_readchar()`
: The calling task is trying to read user input. If input is available you can return it immediately, but if there is no input the task should be blocked. To check for input, call `getch()`. This function will return `ERR` if no input is available, or the code of the typed character if there was one.

### Internals
The starter code includes a partial implementation of some global state for your scheduler library, and a partial implementation of the `task_create` function.
The `task_info_t` struct is designed not to be shared with the user of the system, but to hold the internal state of a task;
this includes its *context*---the state required to return to the running task---and other information about what the task is waiting for or whether it has exited.
There is an array of these `task_info_t` structures called `tasks`;
the value returned in a `task_t` is an index into this array.
You are free to change this structure, but you should not need to unless you decide to do the optional challenge at the end of this lab.

The `task_create` function shows how you can set up a `context_t` to execute a function.
To do this, you first pre-populate the context with information from the current task, set up a stack for the new task, and then use the `makecontext` function to point the context to a function with some number of parameters (zero in our case).

The `makecontext` function sets up a context, but does not run it.
To do that, you will need to use the `swapcontext` function.
You simply call `swapcontext(&context1, &context2)`, which will save the currently-executing task to `context1`, then load in and begin executing from `context2`.
One peculiar feature of this function is that it *sort of* returns.
The processor will begin executing in `context2`, but if you later swap back to `context` it will resume by returning from this call to `swapcontext`.
Your scheduler code will need to use `swapcontext` to jump from a newly-blocked or exited task to the next task you would like to execute.

### Scheduler Requirements
You should implement a round-robin scheduler for this lab.
That means if task 3 blocks, the first candidate task your scheduler should consider is task 4;
if task 4 is blocked, move on to task 5, task 6, and so on.
You do *not* need to reorder tasks to track the order in which they unblocked;
simply cycle through them in the order they were first created, starting with the next task in the sequence after the previously executed one.

## Testing
As you implement your scheduler, you will want to test it.
The `worm` game relies on all the functions in your scheduler, but there are some simpler tests in the `tests` directory.
We will look at each test in class, but you should also refer to the test code to see which scheduler functions it relies on.
I recommend starting your implementation with `task_create`, `task_wait`, and `task_sleep`, since these are easier to test than user input.

## Optional Challenge: Unlimited Tasks
One drawback of the implementation this lab starts with is that there is a hard upper limit on the number of tasks any program can create.
That limit applies to both currently-executing tasks and any tasks created in the past.
If you would like an additional challenge, change your implementation so it supports an arbitrary number of tasks.
That means you will not be able to use a fixed-size global array to track tasks, and you should have some provision for freeing memory allocated to hold task information once a task has exited and "joined" with another task that called `task_wait`.
This will almost certainly require that you change the underlying type of a `task_t`;
think carefully about what kind of type will best allow you to locate a task's information structure without imposing the upper limit in the original implementation.
