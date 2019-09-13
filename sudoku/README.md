## Overview
For this lab, you will write a program that uses the GPU to solve sudoku puzzles.
These puzzles are an interesting example of a constraint-satisfaction problem, which is a general approach to solving a wide variety of problems.
This computation also lends itself well to the GPU: we have a lot of available parallelism within each puzzle, and you will solve many puzzles simultaneously.
We'll spend some time at the start of lab to discuss the rules of sudoku.
We will also discuss strategies for parallelizing sudoku solving and select the approach that will work best on the GPU.

### Acknowlegements
Today's lab was inspired by Peter Norvig's post [Solving Every Sudoku Puzzle](http://norvig.com/sudoku.html){:target="_blank"}.
We will use a technique similar to Norvig's sudoku solver, though you will see that a GPU-based solver is ***much*** faster than Norvig's Python implementation.

The inputs for this lab were obtained from [Kaggle](https://www.kaggle.com/bryanpark/sudoku/kernels){:target="_blank"}, and are licensed CC-0.

## Introduction to Sudoku
If you have never completed a sudoku before, or you are unsure of the rules, you should quickly read through Sudoku Dragon's [Introduction to Sudoku](http://www.sudokudragon.com/sudoku.htm){:target="_blank"}.
I will review the rules below;
even if you already know them, make sure to read carefully so we can agree on terminology that will be important when we discuss solving strategies.

A sudoku board is a 9x9 grid.
Each cell in the grid is either blank, or holds a digit from 1--9 (inclusive.
Starting boards will have a few cells filled in, and your job is to fill in all of the cells.

There are constraints on what values can appear in each cell:

Row Constraints
: Every row in the completed sudoku board must have one copy of each digit 1--9.
  There must be exactly one "1" in each row, one "2" in each row, and so on.
  
Column Constraints
: Every column in the completed sudoku board must have exacly one copy of each
  digit from 1--9.

Region Constraints
: The board is also divided into square regions of nine cells. Each region must
  have exactly one copy of each digit from 1--9.

These constraints are sufficient to limit a sudoku board to just one possible solution;
your task when solving the sudoku is to fill in every cell without violating any constraints.
Creating starting boards that have exactly one possible solution is actually an interesting challenge, both computationally and theoretically;
if you are interested, you can read more about the mathematics of sudoku on [Wikipedia](https://en.wikipedia.org/wiki/Mathematics_of_Sudoku){:target="_blank"}.

There are many strategies that humans use to solve sudoku, but we'll focus on one specific strategy that we can implement on a computer.

## Sudoku Solving Strategies
You can read about some sudoku solving strategies meant for humans on the Sudoku Dragon [strategies](http://www.sudokudragon.com/sudokustrategy.htm){:target="_blank"} page.
While these strategies work well for people, they are designed to minimize the amount of information you have to keep track of;
learning these strategies is learning to recognize patterns and act on them without having to remember something about 81 different cells on the board.

For a computer, remembering detailed information is not a challenge.
What would be challenging is correctly implementing a system to recognize these patterns and act on them without introducing errors.
Instead of using these human-focused strategies, we will solve sudoku using *constraint propagation*.

### Constraint Propagation
All 81 cells of a sudoku board will eventually hold one value, but when the board is incomplete you can think of each cell as a set of possible values.
A blank cells can be any digit from 1--9, and a filled-in cell can be only the digit that is currently placed there.

If a cell is filled in with a fixed value that tells us something about the other cells in the same row, column, or region;
For example if cell Aa (the top left corner) is a "1", we know that the other cells in row A, column a, and the upper-left region cannot be "1".
We can propagate this constraint by removing "1" from the set of possible values for all these other cells.

If we repeat this process for all cells on the board, we will likely discover the correct values of some other cells.
These values then place further constraints in their rows, columns, and regions, which we can propagate.

We continue this process of propagating constraints until the board *converges*.
If propagating a series of constraints does not change any cells' values we can no longer make progress.
For many sudoku, this will mean that the puzzle has been solved.
It is possible, however, that the puzzle is not yet solved;
this can happen in a puzzle where two different values could go in two cells that appear in the same row, but the surrounding cells don't tell us which value goes in either place.
Human sudoku solvers use more sophisticated strategies to move past this point, but most computational sudoku solvers will simply try one assignment of values, backtrack if the assignment is found to be invalid, and then try the alternative assignment.

For this lab, you do not need to move past the constraint propagation phase;
you will be able to solve the vast majority of sudoku in our input set using only constraint propagation.
Adding backtracking to finish solving the remaining sudoku is an good challenge if you find you have additional time for this lab.

## Starter Code
Download and extract the starter code archive.
Nearly all of the code provided appears in `sudoku.cu`.
The provided code reads a set of sudoku from an input file and passes them on to be solved.
The starter code well read `BATCH_SIZE` sudoku boards at once, then invoke the `solve_boards` function, which you will need to write.
By default, sudoku will be solved in batches of 128, though you will need to experiment with changing the batch size later in the lab.

You should run the starter code to make sure everything builds correctly on your account;
to do so, run `make`.
The first time you run `make`, the input archive will be extracted.
Assuming there are no build errors, you should have five input files: `tiny.csv`, `small.csv`, `medium.csv`, `large.csv`, and `huge.csv`.
These files contain one hundred, one thousand, ten thousand, one hundred thousand, and one million sudoku boards, respectively.

Run the starter code and verify that you get roughly the same output as below:

```shell
$ ./sudoku inputs/huge.csv
Boards: 1000000
Boards Solved: 0
Errors: 0
Total Solving Time: 0ms
Solving Rate: 0.00 sudoku/second
```

The starter code includes a check for each solved board to make sure you do not place any incorrect constraints in the board;
the value for "Errors" should always be zero.
You will not be able to solve every sudoku in this set using constraint propagation, but if your implementation is correct you should solve:

1. 100/100 for the tiny set
2. 1000/1000 for the small set
3. 9997/10000 for the medium set
4. 99959/100000 for the large set
5. 999671/1000000 for the huge set

The performance of your solution will depend on how you implement your solver, but my solution solves roughly 240,000 sudoku per second on the huge input after adjusting the batch size.

## Tasks
We will begin the lab by discussing parallelization strategies.
Once we have done so, complete the following tasks for this lab:

1. Read *all* of the comments in the starter code, and skim over the code. There are utility functions that help with the encoding of the board state, but you should understand how these work before moving on.

2. Write a CUDA kernel that will run for each cell in a sudoku board. For now it should do nothing. Invoke that kernel in `solve_boards` to run a thread for every cell in a board, and a block for every board in the batch. You will find it useful to use the `dim3` type for CUDA kernel invocation; this allows you to fill in both `threadIdx.x` and `threadIdx.y` to match the column and row of the cell on a board. You will need to copy the entire batch of boards to the GPU before running the kernel, and copy the boards back off the GPU when the kernel is finished.

3. Implement constraint propagation for your sudoku boards. This will require a loop in your kernel that runs as long as any thread in the block updated constraints in a way that changed the board state. 
   The easiest way to do this is using the `__syncthreads_count(int predicate)` function.
   Each thread passes in some value for the `predicate` parameter, and `__syncthreads_count` returns the number of non-zero predicates.
   Use a local variable in each thread to keep track of whether the constraints changed, and pass this value to `__syncthreads_count`.
   If `__syncthreads_count` returns a non-zero value, at least one cell changed and the kernel should continue propagating constraints.

4. Once you have your solver working, adjust the value of `BATCH_SIZE`. What does increasing or decreasing `BATCH_SIZE` do to the solver's performance? Why? Find a setting that maximizes your solver's performance on the machines in the lab.
