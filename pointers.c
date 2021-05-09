// 1) result == 4
// 2) Forgot the "&" sign before the array when defining the array_ptr
// 3) Size of excercise
// 4) Nothing happens
// 5) Nothing happens

#include <stdio.h>

void e3()
{
    long double ld;
    char dinosaur[5];
    printf("Size of long double: %i\n", sizeof(ld));
    printf("Size of dinosaur[5]: %i\n", sizeof(dinosaur));
}

void e4()
{
    int* null_ptr = NULL;
    (*null_ptr) = 42;
    printf("Null ptr value: %i\n", (*null_ptr));
}

void e5()
{
    int array[2] = {2, 3};
    int* array_ptr = array;
    *(array_ptr + sizeof((*array_ptr))) = 2;
}

void memory_copy(void* to, void* from, int n)
{
    char* cto = to;
    char* cfrom = from;

    for (int i = 0; i < n; i++)
    {
        *(cto + i) = *(cfrom + i);
    }
}

void swap(char* a, char* b)
{
    char temp = *a;
    *a = *b;
    *b = temp;
}

void permutations_rec(char* string, int n)
{
    if (n == 1)
    {
        printf("%s\n", string);
    }
    else
    {
        permutations_rec(string, n - 1);
        for (int i = 1; i < n; i++)
        {
            char* a = string + strlen(string) - n;
            char* b = string + strlen(string) - n + i;
            swap(a, b);
            permutations_rec(string, n - 1);
            swap(a, b);
        }
    }
}

void permutations(char* string)
{
    permutations_rec(string, strlen(string));
}

int main()
{
    // e3();
    // e4();
    // e5();

    permutations("abc");
}