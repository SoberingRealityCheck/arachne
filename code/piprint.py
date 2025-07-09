from termcolor import colored

def piprint(text):
    """
    Print text with a specific color and style.
    
    Args:
        text (str): The text to print.
    """
    print(colored(text, 'green', attrs=['bold']))