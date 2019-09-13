#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "scheduler.h"

void task1_fn() {
  printf("Task 1: Sleeping for two seconds\n");
  task_sleep(2000);
  printf("Task 1: Woke up\n");
}

void task2_fn() {
  printf("Task 2: Sleeping for one second\n");
  task_sleep(1000);
  printf("Task 2: Woke up\n");
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

