#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// I wanted to make this 68419 but did not want to insult her majesty by calling her tribe long
// #define NUMBER_OF_MEMBERS 50000

unsigned char tribe [50000];

char * readFile(FILE * file)
{
    if (file == NULL)
    {
        perror("Error when opening runes.\n");
        exit(EXIT_FAILURE);
    }

    char * contents;

    fseek (file, 0, SEEK_END);
    int length = ftell (file);
    fseek (file, 0, SEEK_SET);

    contents = malloc (length);
    fread(contents, 1, length, file);
    fclose(file);

    return contents;
}

void checkParenBalance(char * runes)
{
    int parenCount = 0;
    while (*runes != '\0')
    {
        switch (*runes)
        {
            case '(': parenCount++; break;
            case ')': parenCount--; break;
            default: break;
        }

        if (parenCount < 0)
        {
            printf("Unbalanced Parenthesis\n");
            exit(EXIT_FAILURE);
        }

        runes++;
    }

    if (parenCount != 0)
    {
        printf("Unbalanced Parenthesis\n");
        exit(EXIT_FAILURE);
    }
}

char * runeSecretary(char * runes)
{
    // After The Secretary That Never Get's Credit Optimizes The Runes
    char * ATSTNGCOTRunes = malloc(2 * strlen(runes));
    char * ATSTNGCOTRunesStart = ATSTNGCOTRunes;

    while (*runes != '\0')
    {
        // printf("\nReading %c\n", *runes);
        unsigned char largeness = 1; // Can hold up to 255

        switch (*runes)
        {
            case '<':
                *ATSTNGCOTRunes = '<';
                ATSTNGCOTRunes++;

                while (*(runes + 1) == '<')
                {
                    largeness++;
                    runes++;
                }

                *ATSTNGCOTRunes = largeness;

                break;
            case '>':
                *ATSTNGCOTRunes = '>';
                ATSTNGCOTRunes++;

                while (*(runes + 1) == '>')
                {
                    largeness++;
                    runes++;
                }

                *ATSTNGCOTRunes = largeness;
                break;
            case '+':
                *ATSTNGCOTRunes = '+';
                ATSTNGCOTRunes++;

                while (*(runes + 1) == '+')
                {
                    largeness++;
                    runes++;
                }

                *ATSTNGCOTRunes = largeness;

                break;
            case '-':
                *ATSTNGCOTRunes = '-';
                ATSTNGCOTRunes++;

                while (*(runes + 1) == '-')
                {
                    largeness++;
                    runes++;
                }

                *ATSTNGCOTRunes = largeness;

                break;
            case '(': *ATSTNGCOTRunes = '('; break;
            case ')': *ATSTNGCOTRunes = ')'; break;
            case '*': *ATSTNGCOTRunes = '*'; break;
            default: ATSTNGCOTRunes--;
        }

        runes++;
        ATSTNGCOTRunes++;
    }

    *ATSTNGCOTRunes = '\0';

    return ATSTNGCOTRunesStart;
}

void printRunes(char * runes)
{
    while (*runes != '\0')
    {
        printf("%c", *runes);
        runes++;
    }
    printf("\n");
}

unsigned int moveLeft(unsigned int divinePointress, unsigned char largeness)
{
    if ((int) (divinePointress - largeness) < 0)
    {
        printf("Sorry for my intrusion Miss Divine Pointress, but you must stay in front of your members\n");
        exit(EXIT_FAILURE);
    }

    return divinePointress - largeness;
}


unsigned int moveRight(unsigned int divinePointress, unsigned char largeness)
{
    if (divinePointress + largeness > 50000)
    {
        printf("Sorry for my intrusion Miss Divine Pointress, but I must save you from walking too far to the right\n");
        exit(EXIT_FAILURE);
    }

    // printf("Moved Right\n");
    return divinePointress + largeness;
}

void addPebble(unsigned int divinePointress, unsigned char largeness)
{
    tribe[divinePointress] += largeness;
    // printf("Added Pebble\n");
}

void removePebble(unsigned int divinePointress, unsigned char largeness)
{
    tribe[divinePointress] -= largeness;
    // printf("Removed Pebble\n");
}

char * fCheckPebbles(unsigned int divinePointress, char * runes)
{
    if (tribe[divinePointress] == 0)
    {
        int parenCount = 1;

        while (parenCount > 0)
        {
            runes++;

            switch (*runes)
            {
                case '(': parenCount++; break;
                case ')': parenCount--; break;
                default: break;
            }
        }
        // printf("Jumped Forward\n");
    }

    // printf("Checked Pebbles to Jump Forward\n");

    return runes;
}

char * bCheckPebbles(unsigned int divinePointress, char * runes)
{
    if (tribe[divinePointress] != 0)
    {
        int parenCount = 1;

        while (parenCount > 0)
        {
            runes--;

            switch (*runes)
            {
                case ')': parenCount++; break;
                case '(': parenCount--; break;
                default: break;
            }
        }
            
        // printf("Jumped Back\n");
    }

    // printf("Checked Pebbles to Jump Back\n");

    return runes;
}

void yell(unsigned int divinePointress)
{
    printf("%c", tribe[divinePointress]);
    // printf("Yelled\n");
}

void readRunes(char * runes)
{
    unsigned int divinePointress = 0;

    while (*runes != '\0')
    {
        // printf("\nReading %c\n", *runes);

        switch (*runes)
        {
            case '<': divinePointress = moveLeft(divinePointress, *(++runes)); break;
            case '>': divinePointress = moveRight(divinePointress, *(++runes)); break;
            case '+': addPebble(divinePointress, *(++runes)); break;
            case '-': removePebble(divinePointress, *(++runes)); break;
            case '(': runes = fCheckPebbles(divinePointress, runes); break;
            case ')': runes = bCheckPebbles(divinePointress, runes); break;
            case '*': yell(divinePointress); break;
            default: break;
        }

        runes++;
    }

    // printf("\n=== Finished Reading Runes! ===\n");
}

int main(int argc, char **argv)
{
    FILE * runesFile = fopen(argv[1], "r");

    // clock_t begin = clock();
    
    char * runes = readFile(runesFile);

    checkParenBalance(runes);

    // printRunes(runes);

    runes = runeSecretary(runes);

    // printRunes(runes);
    
    readRunes(runes);

    // clock_t end = clock();
    // double timeSpent = (double)(end - begin) / CLOCKS_PER_SEC;
    // printf("Time Spent: %f\n", timeSpent);

    return(0);
}