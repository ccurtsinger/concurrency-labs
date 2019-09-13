CC := clang
CFLAGS := -g -Wall -Wno-deprecated-declarations -Werror

TESTS := test1 test2 test3 test4

all: $(TESTS)

clean:
	rm -rf $(TESTS) *.dSYM

test%: test%.c ../scheduler.c ../scheduler.h ../util.c ../util.h
	$(CC) $(CFLAGS) -I.. -o $@ $< ../scheduler.c ../util.c -lncurses
