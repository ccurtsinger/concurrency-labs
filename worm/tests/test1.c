#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "scheduler.h"

void task1_fn() {
  printf("I am task 1.\n");
}

void task2_fn() {
  printf("I am task 2.\n");
}

int main() {
  scheduler_init();
  
  task_t task1;
  task_t task2;
  
  task_create(&task1, task1_fn);
  task_create(&task2, task2_fn);
  
  task_wait(task1);
  task_wait(task2);
  
  printf("All done!\n");
  
  return 0;
}

