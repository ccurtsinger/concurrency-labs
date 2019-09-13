#ifndef QUEUE_H
#define QUEUE_H

#include <stdbool.h>

typedef struct my_queue {
  // Add fields to your queue type here
  int placeholder;
} my_queue_t;

// Initialize a queue
void queue_init(my_queue_t* queue);

// Destroy a queue
void queue_destroy(my_queue_t* queue);

// Put an element at the end of a queue
void queue_put(my_queue_t* queue, int element);

// Chekc if a queue is empty
bool queue_empty(my_queue_t* queue);

// Take an element off the front of a queue
int queue_take(my_queue_t* queue);

#endif
