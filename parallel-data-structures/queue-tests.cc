#include <gtest/gtest.h>

#include "queue.hh"

/****** Queue Invariants ******/

// TODO: Write invariants here. See stack invariants for reference.

/****** Begin Tests ******/

// Basic queue functionality
TEST(QueueTest, BasicQueueOps) {
  my_queue_t q;
  queue_init(&q);
  
  // Make sure the queue is empty
  ASSERT_TRUE(queue_empty(&q));
  
  // Make sure taking from the queue returns -1
  ASSERT_EQ(-1, queue_take(&q));
  
  // Add some items to the queue
  queue_put(&q, 1);
  queue_put(&q, 2);
  queue_put(&q, 3);
  
  // Make sure the queue is not empty
  ASSERT_FALSE(queue_empty(&q));
  
  // Take the values from the queue and check them
  ASSERT_EQ(1, queue_take(&q));
  ASSERT_EQ(2, queue_take(&q));
  ASSERT_EQ(3, queue_take(&q));
  
  // Make sure the queue is empty
  ASSERT_TRUE(queue_empty(&q));
  
  // Clean up
  queue_destroy(&q);
}
