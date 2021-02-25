import os, sys, random

WORK_DIR = 'work_dir/training-monolingual'
TEST_DIR = 'work_dir/output'
FILE_PREFIX = 'news.2011.{}.shuffled'
LANG = ['cs', 'de', 'en', 'es', 'fr']
# LANG = ['en']
DO_FILTER = False
DO_GEN_TEST = True
THRESHOLD = 0.05

if __name__ == '__main__':

    if DO_FILTER:
        for lang in LANG:
            input = open(os.path.join(WORK_DIR, FILE_PREFIX.format(lang)),
                         'r', encoding='utf-8')
            output = open(os.path.join(WORK_DIR, f'{lang}.filtered'),
                          'w', encoding='utf-8')
            test_output = open(os.path.join(TEST_DIR, f'{lang}.test'),
                               'w', encoding='utf-8')
            for line in input:
                if random.random() < THRESHOLD:
                    test_output.write(line)
                else:
                    output.write(line)
            input.close()
            output.close()
            test_output.close()

    if DO_GEN_TEST:
        test_input = open(os.path.join(TEST_DIR, 'input.txt'),
                          'w', encoding='utf-8')
        test_answer = open(os.path.join(TEST_DIR, 'answer.txt'),
                           'w', encoding='utf-8')
        for lang in LANG:
            test_whole = open(os.path.join(TEST_DIR, f'{lang}.test'),
                              'r', encoding='utf-8')
            for line in test_whole:
                strip = line.strip()
                if len(strip) > 2:
                    idx = random.randint(1, len(strip) - 1)
                    test_input.write(strip[:idx] + '\n')
                    test_answer.write(strip[idx] + '\n')
            test_whole.close()

        test_input.close()
        test_answer.close()
