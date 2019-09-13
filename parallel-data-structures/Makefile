CXX := clang++
CXXFLAGS := -g -Wall -Werror
GTEST_FLAGS :=  -isystem gtest -isystem gtest/include gtest/src/gtest-all.cc gtest/src/gtest_main.cc

all: stack-tests queue-tests dict-tests

clean:
	rm -rf stack-tests stack-tests.dSYM queue-tests queue-tests.dSYM dict-tests dict-tests.dSYM

stack-tests: stack-tests.cc stack.cc stack.hh gtest
	$(CXX) $(CXXFLAGS) -o stack-tests $(GTEST_FLAGS) stack-tests.cc stack.cc -lpthread

queue-tests: queue-tests.cc queue.cc queue.hh gtest
	$(CXX) $(CXXFLAGS) -o queue-tests $(GTEST_FLAGS) queue-tests.cc queue.cc -lpthread

dict-tests: dict-tests.cc dict.cc dict.hh gtest
	$(CXX) $(CXXFLAGS) -o dict-tests $(GTEST_FLAGS) dict-tests.cc dict.cc -lpthread

gtest:
	wget https://github.com/google/googletest/archive/release-1.7.0.tar.gz
	tar xzf release-1.7.0.tar.gz
	rm release-1.7.0.tar.gz
	mv googletest-release-1.7.0 gtest
