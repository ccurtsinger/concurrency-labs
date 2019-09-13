#include <curses.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include "scheduler.h"

void task1_fn() {
  while(true) {
    printw("Tick\n");
    task_sleep(1000);
  }
}

void task2_fn() {
  // Wait for the user to type four characters
  for(int i=0; i<4; i++) {
    int c = task_readchar();
    printw("You typed %c\n", c);
  }
}

int main() {
  // Initialize the ncurses window
  WINDOW* mainwin = initscr();
  if(mainwin == NULL) {
    fprintf(stderr, "Error initializing ncurses.\n");
    exit(2);
  }
  
  // Set up input with ncurses
  noecho();
  keypad(mainwin, true);
  nodelay(mainwin, true);
  
  // Begin the test
  scheduler_init();
  
  task_t task1;
  task_t task2;
  
  task_create(&task1, task1_fn);
  task_create(&task2, task2_fn);
  
  // Only wait for task 2 to finish, since task 1 runs forever
  task_wait(task2);
  
  // Clean up window
  delwin(mainwin);
  endwin();
  
  return 0;
}

