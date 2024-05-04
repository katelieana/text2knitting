import re
from typing import List
# from itertools import batched
import argparse

test_text = 'test helloworldsofwater how are you doing today 2024-05-03?'
expected_test_output = 'p4 k1 p5 \np7 k1 p2 \np4 k1 p3 k1 p1 \np2 k1 p3 k1 p3 \np2 k1 p5 k1 p1 \np1 k1 p2 k2 p2 k1 p1 \np3 k2 p2 k2 p1 \np4 k2 p2 k2 \np3 k2 '

def text_to_stitches(text: str, punc_spaces: int=2, zero_as_punctuation: bool=True):
    stitches = ''
    
    for char in text.lower():
        
        if re.match(r'\s+', char):
            # all whitespace characters are 1:1 knit stitches
            stitches += 'k'
            
        elif re.match(r'\d+', char):
            # treates 0 as punctuation if flag is set
            if zero_as_punctuation and char == '0':
                    stitches += 'p'*punc_spaces + 'k'

            # all other digits are 1:1 purl stitches
            stitches += 'p' * int(char) + 'k'
            
        elif re.match(r'\w+', char):
            # all letters are 1:1 purl stitches
            stitches += 'p'

        else:
            # catchall, everything else is treated as punctuation and is 1:2 knit stitches
            stitches += 'k' * punc_spaces
            
    return stitches

def fix_long_repeats(string: str, 
                     max_purl_run_len: int=12, 
                     max_knit_run_len: int=2, 
                     purl_run_spacer_char: str='k'): 
    
    # Fix long purl runs
    pattern = r'(p)\1{' + str(max_purl_run_len) + r',}'
    matches = re.finditer(pattern, string)
    start_ixs = [match.start() for match in matches]
    
    for ix in start_ixs:
        string = string[:ix+max_purl_run_len] + purl_run_spacer_char + string[ix+max_purl_run_len:]
        
    # Fix long knit runs
    pattern = r'k{' + str(max_knit_run_len+1) + r',}'
    string = re.sub(pattern, 'k' * int(max_knit_run_len), string)
        
    return string

def consolidate_sequence_into_pattern(sequence: str):
    consolidated = ''
    current = sequence[0]
    count = 1

    for stitch in sequence[1:]:
        if stitch == current:
            count += 1
        else:
            consolidated += (current + str(count) + ' ')
            current = stitch
            count = 1
    
    consolidated += (current + str(count) + ' ')
    
    return consolidated

def consolidate_sequences(sequences: List[str]):
    pattern = []
    for s in sequences:
        pattern.append(consolidate_sequence_into_pattern(s))
        
    return '\n'.join(pattern)

def break_sequence(sequence: str, grouping_size: int=10): 
    return [sequence[i:i+grouping_size] for i in range(0, len(sequence), grouping_size)]

# itertools version; only avalaible in python 3.10 or later 
# def make_readable_pattern(sequence: str, block_size: int=10, blocks_in_line: int=3): 

#     blocks = [consolidate_sequence_into_pattern(''.join(b)) for b in batched(sequence, block_size)]
#     lines = ['\t'.join((l)) for l in batched(blocks, blocks_in_line)]
    
#     return '\n'.join(lines)

def make_readable_pattern(sequence: str, single_line:bool = False, block_size: int=10, blocks_in_line: int=3): 
    
    if single_line:
        block_size = len(sequence)
        blocks_in_line = 1
            
    blocks = [consolidate_sequence_into_pattern(b) for b in break_sequence(sequence, block_size)]

    lines = []
    while len(blocks) > 0:
        lines.append('\t'.join(blocks[:blocks_in_line]))
        blocks = blocks[blocks_in_line:]
    
    return '\n'.join(lines)

def get_cleaned_stitch_sequence(text: str, 
                                max_purl_run_len: int=12, 
                                max_knit_run_len: int=2, 
                                purl_run_spacer_char: str='k', 
                                punctuation_mapping: int=2, 
                                treat_zero_as_punctuation: bool=True
                                ):
                
    stitches = text_to_stitches(text, 
                                punc_spaces=punctuation_mapping, 
                                zero_as_punctuation=treat_zero_as_punctuation
                                )
    
    stitches = fix_long_repeats(stitches, 
                                max_purl_run_len=max_purl_run_len, 
                                max_knit_run_len=max_knit_run_len,
                                purl_run_spacer_char=purl_run_spacer_char
                                )
    
    return stitches

def text_to_knitting(text: str, 
                     single_line: bool=False,
                     stitches_per_block: int=10, 
                     blocks_per_line: int=3,
                     max_purl_run_len: int=12, 
                     max_knit_run_len: int=2, 
                     purl_run_spacer_char: str='k', 
                     punctuation_mapping: int=2, 
                     treat_zero_as_punctuation: bool=True
                     ):
    
    stitches = text_to_stitches(text, 
                                punc_spaces=punctuation_mapping, 
                                zero_as_punctuation=treat_zero_as_punctuation
                                )
    
    stitches = fix_long_repeats(stitches, 
                                max_purl_run_len=max_purl_run_len, 
                                max_knit_run_len=max_knit_run_len,
                                purl_run_spacer_char=purl_run_spacer_char
                                )
        
    pattern = make_readable_pattern(stitches, 
                                    single_line=single_line,
                                    block_size=stitches_per_block, 
                                    blocks_in_line=blocks_per_line
                                    )
    
    return pattern


def divide_original_text(text: str, grouping_size: int=10):
    return '\n'.join([text[i:i+grouping_size] for i in range(0, len(text), grouping_size)]).strip()




class Bibliophile:
    def __init__(self, 
                 text: str, 
                 stitches_per_block: int=10, 
                 blocks_per_line: int=1,
                 max_purl_run_len: int=12, 
                 max_knit_run_len: int=2, 
                 purl_run_spacer_char: str='k', 
                 punctuation_mapping: int=2, 
                 treat_zero_as_punctuation: bool=True
                 ):
        
        self.text = text
        self.stitches_per_block = stitches_per_block
        self.blocks_per_line = blocks_per_line
        self.max_purl_run_len = max_purl_run_len
        self.max_knit_run_len = max_knit_run_len
        self.purl_run_spacer_char = purl_run_spacer_char
        self.punctuation_mapping = punctuation_mapping
        self.treat_zero_as_punctuation = treat_zero_as_punctuation
        
        self.current_index = 0
        
        self.called_stitch_counts = []
        
        self.stitches = get_cleaned_stitch_sequence(text, 
                                                   max_purl_run_len=max_purl_run_len, 
                                                   max_knit_run_len=max_knit_run_len, 
                                                   purl_run_spacer_char=purl_run_spacer_char, 
                                                   punctuation_mapping=punctuation_mapping, 
                                                   treat_zero_as_punctuation=treat_zero_as_punctuation
                                                   )
        
        self.pattern = text_to_knitting(text, 
                                        stitches_per_block=stitches_per_block, 
                                        blocks_per_line=blocks_per_line, 
                                        max_purl_run_len=max_purl_run_len, 
                                        max_knit_run_len=max_knit_run_len, 
                                        purl_run_spacer_char=purl_run_spacer_char, 
                                        punctuation_mapping=punctuation_mapping, 
                                        treat_zero_as_punctuation=treat_zero_as_punctuation
                                        )
        
        
    def __str__(self):
        return self.pattern
    
    def __repr__(self):
        return self.pattern
    
    def __getitem__(self, index):
        return self.pattern[index]
    
    def __len__(self):
        return len(self.stitches)
    
    def __iter__(self):
        return iter(self.stitches)
    
    def __next__(self):
        return next(self.stitches)
    
    def __call__(self, index):
        return self.stitches[index]
    
    def get_stitches_at_index(self, start, end, single_line=True):
        return make_readable_pattern(self.stitches[start:end], single_line=single_line)
    
    def get_next_block_row(self):
        return next(self.pattern_rows)
    
    def reset_index(self):
        self.current_index = 0
        
    def set_index(self, index):
        self.current_index = index
        
    def get_index(self):
        return self.current_index
        
    def get_called_stitch_counts(self):
        return self.called_stitch_counts
            
    def reset_called_stitch_counts(self):
        self.called_stitch_counts = []
        
    def get_pattern_as_string(self):
        return self.pattern
    
    def print_pattern(self):
        print(self.pattern)
        
    def generate_pattern(self, 
                         stitches_per_block=None, 
                         blocks_per_line=None, 
                         max_purl_run_len=None, 
                         max_knit_run_len=None, 
                         purl_run_spacer_char=None, 
                         punctuation_mapping=None, 
                         treat_zero_as_punctuation=None
                         ):
        """
        Generate a knitting pattern from text
        Rules for text:pattern translation are as outlined in the pattern
        
        Args:
            stitches_per_block (int): Number of stitches per grouping in final pattern
            blocks_per_line (int): Number of stitch blocks per line, separated by tabs
            max_purl_run_len (int): Maximum length of purl run before inserting spacer
            max_knit_run_len (int): Maximum length of knit run before shortening
            purl_run_spacer_char (str): Character to insert between purl runs
            punctuation_mapping (int): Number of knit stitches to use for punctuation
            treat_zero_as_punctuation (bool): Treat 0 as punctuation (ie, translate it as 2 knits )
        
        """
        
        stitches_per_block = self.stitches_per_block if stitches_per_block is None else stitches_per_block
        blocks_per_line = self.blocks_per_line if blocks_per_line is None else blocks_per_line
        max_purl_run_len = self.max_purl_run_len if max_purl_run_len is None else max_purl_run_len
        max_knit_run_len = self.max_knit_run_len if max_knit_run_len is None else max_knit_run_len
        purl_run_spacer_char = self.purl_run_spacer_char if purl_run_spacer_char is None else purl_run_spacer_char
        punctuation_mapping = self.punctuation_mapping if punctuation_mapping is None else punctuation_mapping
        treat_zero_as_punctuation = self.treat_zero_as_punctuation if treat_zero_as_punctuation is None else treat_zero_as_punctuation
        
        pattern = text_to_knitting(self.text, 
                                   stitches_per_block=stitches_per_block, 
                                   blocks_per_line=blocks_per_line, 
                                   max_purl_run_len=max_purl_run_len, 
                                   max_knit_run_len=max_knit_run_len, 
                                   purl_run_spacer_char=purl_run_spacer_char, 
                                   punctuation_mapping=punctuation_mapping, 
                                   treat_zero_as_punctuation=treat_zero_as_punctuation
                                   )
        return pattern
    
    def get_next_n_stitches(self, 
                            n_stitches: int, 
                            single_line: bool=False, 
                            stitches_per_block=None, 
                            blocks_per_line=None,
                            ):
                   
        if single_line:
            stitches_per_block = n_stitches
            blocks_per_line = 1
            
        else:
            stitches_per_block = self.stitches_per_block if stitches_per_block is None else stitches_per_block
            blocks_per_line = self.blocks_per_line if blocks_per_line is None else blocks_per_line
        
        working_row = self.stitches[self.current_index:self.current_index+n_stitches]
        self.current_index += n_stitches
        self.called_stitch_counts.append(n_stitches)
        
        pattern = make_readable_pattern(working_row, 
                                        block_size=stitches_per_block, 
                                        blocks_in_line=blocks_per_line
                                        )
        return pattern
        
    def get_next_T_sequences(self, 
                        stitch_counts: List[int], 
                        start_row_num: int=1, 
                        count_rows_by: int=2, 
                        seq_per_row: int=1, 
                        single_line: bool=True, 
                        stitches_per_block: int=None,
                        blocks_per_line: int=None, 
                        ):
        
        """
        Generate the next series of 'T' sequences in the pattern, given the expected stitch counts. 
        The starting stitch (word) will automatically be advanced each time this function is called based on the last used index (stitch). 
        
        Args:
            stitch_counts (List[int]): List of stitch counts for each sequence
            start_row_num (int): Starting row number for the pattern
            count_rows_by (int): Number of rows to count by
            seq_per_row (int): Number of sequences to print per row
            single_line (bool): Print each sequence on a single line when True, otherwise follow the blocking rules defined by stitches_per_block and blocks_per_line. When True, this will override the stitches_per_block and blocks_per_line arguments
            stitches_per_block (int): Number of stitches per grouping in final pattern
            blocks_per_line (int): Number of stitch blocks per line, separated by tabs
    
        """
        
        pattern = []
        seq_num = 0
        
        for i in range(len(stitch_counts)):
            if seq_num % seq_per_row == 0:
                pattern.append('\n----- \nROW ' + str(start_row_num) + '\n')
                start_row_num += count_rows_by
                seq_num = 0
            
            pattern.append('\nT' + str(seq_num+1) + ': (' + str(stitch_counts[i]) + ') \n')
            pattern.append(self.get_next_n_stitches(n_stitches=stitch_counts[i], 
                                                    single_line=single_line, 
                                                    stitches_per_block=stitches_per_block, blocks_per_line=blocks_per_line
                                                    ))
            pattern.append('\n')
            seq_num += 1    
            
        return ''.join(pattern)
    
    
if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Convert text to knitting pattern')
    argparser.add_argument('--text', type=str, default=test_text, help='Text to convert to knitting pattern')
    argparser.add_argument('--stitches_per_block', type=int, default=10, help='Number of stitches per group in final pattern')
    argparser.add_argument('--blocks_per_line', type=int, default=1, help='Number of stitch blocks per line')    
    argparser.add_argument('--max_purl_run_len', type=int, default=12, help='Maximum length of purl run before inserting spacer')
    argparser.add_argument('--max_knit_run_len', type=int, default=2, help='Maximum length of knit run before shortening')
    argparser.add_argument('--purl_run_spacer_char', type=str, default='k', help='Character to insert between purl runs')
    argparser.add_argument('--punctuation_mapping', type=int, default=2, help='Number of knit stitches to use for punctuation')
    argparser.add_argument('--treat_zero_as_punctuation', type=bool, default=True, help='Treat 0 as punctuation')
    
    pattern = text_to_knitting(**vars(argparser.parse_args()))    
        
    assert pattern == expected_test_output, 'Test failed'
    
    print(pattern)

