

from typing import TextIO

from pathlib import Path

#from rich.console import Console

#console = Console()


def remove_comment_blocks(
    path_input_file: str = "text.txt",
    path_output_file: str = "output.txt",
    block_start_with: str = "#",
    ) -> TextIO:

    with open(path_input_file, "r", encoding="utf-8") as input_file:
        with open(path_output_file, "w", encoding="utf-8") as output_file:
            for input_line in input_file:
                if input_line.startswith(block_start_with):
                    writeout = False
                else:
                    writeout = True
                if writeout:
                    output_line = input_line
                    output_file.write(output_line)

        output_file.close()
    input_file.close()

def main():

    file_path = input("Enter input/output file path:")
    input_file = input('Enter input file name:')
    output_file = input('Enter output file name:')
    block_start_with = input('Enter block start pattern:')

    remove_comment_blocks(
        path_input_file = Path(file_path) / input_file,
        path_output_file = Path(file_path) / output_file,
        block_start_with = block_start_with
        )
    
if __name__ == "__main__":
    main()

