CC := clang
CFLAGS := -g -Wall -Wno-deprecated-declarations -Werror

all: worm

clean:
	rm -rf worm worm.dSYM

worm: worm.c util.c util.h scheduler.c scheduler.h
	$(CC) $(CFLAGS) -o worm worm.c util.c scheduler.c -lncurses
