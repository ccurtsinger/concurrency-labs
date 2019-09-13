## Overview

In this lab, you will implement a *password cracker*.
A password cracker is a program that recovers users' passwords from a database of *hashed* passwords.
In operating systems and web applications it is standard practice to *not* store users' passwords in the database of user accounts.
Instead, the application typically runs each user's password through a *hash function* and stores the result.
A hash function must be consistent---every time I run my password through a given hash function I should get the same output---but these functions are meant to work only one way.
I can easily convert passwords to hashed passwords, but it is very difficult to go from hashed passwords back to passwords.
However, sometimes it is possible to recover passwords from hashes; a password cracker performs this feat by searching over the entire space of possible passwords, hashing each one and comparing it to the list of known password hashes.
Any time there is a match we now know the users' original password.

Searching over password hashes is quite difficult for secure hash functions, but luckily there are some *insecure* hash functions that are still widely used.
One of these hash functions is MD5, which is no longer recommended for cryptographic uses.
Your program will receive a list of usernames and MD5-hashed passwords.
It should then search over all possible passwords, hashing each one, until it finds a match for each user's password.
The space of possible passwords is somewhat constrained (see the lab details below), which makes this search process feasible.
Still, there are many candidate passwords to try.
To complete the search in a reasonable amount of time, we will use POSIX threads to perform the search in parallel.

## Part A: Cracking a Single Password
The starter code includes a simple program that runs in two modes: one mode cracks a single hashed password, while the other reads a file containing a list of usernames and hashed passwords, and then cracks all of them.
You should start by implementing the mode to crack a single password.

The first step in running the password cracker in this mode is to generate a hashed password.
We can do this using some standard command line tools:

```shell
$ echo -n "psswrd" | md5sum
63bddf0cbc21d36c8c19808e22784df2
```

Now that we have a hash of our bad six-character password, we can send it to the password cracker.

```shell
$ ./password-cracker single 63bddf0cbc21d36c8c19808e22784df2
psswrd
```

That output is what you should see when you run the unmodified starter code, but it only works for the candidate password "psswrd".
You will need to complete the implementation to crack all potential passwords, but first we need to understand the rules for possible passwords.

### Password Rules
To make our password cracking relatively efficient, we will constrain passwords slightly more than what you'd expect to find for a real password.
Our password cracker will only work for passwords that adhere to the following rules:

1. Passwords are exactly six characters
2. All characters of a password will be lowercase alphabetic characters (a--z)

### Cracking a Password
To actually crack a password, your program will need to perform the following steps:

1. Generate a candidate password
2. Compute the MD5 hash of this candidate password
3. Compare the hash of our candidate password to the provided hash value
4. If the two hashes are equal, output the candidate password and exit. Otherwise go to step 1.

The starter code includes an example call to the `MD5` function that you will find helpful in your implementation.

I *strongly recommend* that you develop an encoding scheme to walk through all possible passwords in a systematic fashion.
Choosing candidate passwords at random could work for short passwords, but six characters each selected from 26 possibilities means there are over 300 million possible passwords to try;
if you choose passwords at random you will almost certainly repeat the same candidate password, which cannot possibly help you find the real password.

## Part B: Cracking a List of Passwords
You will also need to complete the implementation of a *list* mode for the password cracker.
Cracking passwords from a list is more efficient than cracking each password individually because you can hash a candidate password and compare the hash to *every user's hashed password.*

The starter code include the logic to read in a list of users and hashed passwords.
For each user and password, the starter code calls `add_password` to store the username and hashed password in a data structure.
Once all passwords have been read, the starter code calls `crack_password_list`.
You will need to design a `password_set_t` data structure to hold usernames and hashed passwords, implement `init_password_set` and `add_password` to interact with that data structure, and then implement `crack_password_list`, which should crack the passwords stored in a given list.

You may find it useful to reuse some elements of your code from part A, but you should not check each candidate password individually;
that undoes the advantage of cracking passwords from a list.
Make sure you do not break or modify the single password mode of your password cracker.
Both modes must work in your final submission.

Here is an example run of the password cracker in list mode once you have completed your implementation:

```shell
$ ./password-cracker list inputs/input1.txt
jordan passwd
taylor secret
Cracked 2 of 2 passwords.
```

The user/password lines could appear in any order, depending on how you walk through candidate passwords.

This is assuming passwords.txt contains the following values:

```
jordan 76a2173be6393254e72ffa4d6df1030a
taylor 5ebe2294ecd0e0f08eab7690d2a6ee69
```

You should test your implementation on both input files in the `inputs` directory.

## Part C: Cracking Passwords in Parallel
Now that your password cracker can run on a list of users and passwords, you will need to make it faster by dividing the work up across four threads.
To do this, you will use POSIX threads to check candidate passwords in parallel.
This is an instance of a type of problem often referred to as *embarrassingly parallel*.
You can have each thread generate candidate passwords and check their hashes against the database of password hashes without any sort of coordination or dependence between tasks.
The trick here is to divide up the space of candidate passwords over the available threads so they don't duplicate any effort.

You will need to decide how to divide the work and pass that work on to threads.
If your threads make any changes to the data structure as you crack passwords (e.g. by removing an entry once it has been cracked) you will need to include appropriate synchronization to prevent race conditions.
Your program should never produce an error when run with ThreadSanitizer, which checks for unsynchronized modifications to shared values.

As with part B, your implementation may produce output in a different order, and that order may now change between runs;
that is expected.
Just make sure your implementation cracks every password and prints the cracked password with the correct username.

You do not need to preserve your part B implementation; just submit the version of your code with a working parallel password cracker.

### Extra Credit: Password Cracking Competition
When you submit your password cracker to Gradescope it will run against a database of 1000 random passwords.
These passwords will adhere to the password rules listed above, but will change on each run.
The two fastest implementations will receive a 10% extra credit bonus on this lab.
You will be able to see where your implementation ranks once you submit it using Gradescope's leaderboard page.

There are a few rules and implementation details that may be helpful:

1. Your implementation may not use any pre-computed values. The one exception is if you choose to use an alternative MD5 implementation that has a table of a few dozen constants. Alternative MD5 implementations are unlikely to help anyway, since the OpenSSL MD5 implementation is *very* fast.
2. You can only change password-cracker.c; the evaluation will use the default `Makefile` and compilation options.
3. You may only use the POSIX threads (pthreads) API for parallel computation; other parallel libraries like MPI, OpenCL, Cilk, and CUDA are not allowed.
4. The evaluation will run on Gradescope's virtual machine, which has four processors. Your implementation *must* use exactly four threads to perform the password cracking (plus one main thread).
5. I will measure time from the start of the program to when it exits. The program must crack all the passwords before exiting, but you probably do not want to keep trying candidate passwords after cracking the last password.
