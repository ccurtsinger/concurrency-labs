#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "scheduler.h"

void task1_fn() {
  printf("Task 1 will print every two seconds.\n");
  for(int i=0; i<5; i++) {
    printf("Task 1: Tick!\n");
    task_sleep(2000);
  }
  printf("Task 1: Finished.\n");
}

void task2_fn() {
  printf("Task 2 will print every 1.5 seconds.\n");
  for(int i=0; i<5; i++) {
    printf("Task 2: Tock!\n");
    task_sleep(1500);
  }
  printf("Task 2: Finished.\n");
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

